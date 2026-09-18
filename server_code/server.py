from flask import Flask, request, jsonify, Response, send_from_directory, redirect, url_for
import os
import numpy as np
from PIL import Image
import collections
import threading
import time

import tensorflow as tf

from app_config import (
    CORRECTED_DIR, CROP_TEST_DIR, DEFAULT_CONFIG, EXTEND_PROBABILITY_THRESHOLD,
    EXTEND_THRESHOLD, FLAGGED_DIR, IMAGE_RETENTION_SEC, IMG_SIZE, LOCAL_TIMEZONE,
    MIN_OPPOSITE_ACTION_INTERVAL_SEC, MODEL_PATH, OPERATING_END_HOUR,
    OPERATING_START_HOUR, RETRACT_PROBABILITY_THRESHOLD, RETRACT_THRESHOLD,
    SAVE_DIR, SLEEP_DURATION_SEC, VOTE_WINDOW, load_config, load_crop_config,
    local_now, save_config, save_crop_config,
)
from crop_manager import apply_user_crop, validate_crop
from discovery import start_discovery_service
from image_processing import analyze_overexposure, correct_image
from retention import expire_fifo_images, start_retention_worker

app = Flask(__name__)

# ==========================================================
# OVERLAPPING FIFO SLIDING-WINDOW VOTING SYSTEM
# ==========================================================
prediction_history = collections.deque(maxlen=VOTE_WINDOW)
state_lock = threading.Lock()
model_lock = threading.RLock()
process_lock = threading.Lock()
model = None
processing_enabled = True
device_lock = threading.Lock()
test_capture = {
    "request_id": None,
    "status": "idle",
    "filename": None,
    "requested_at": None,
}

rack_state = {
    # Safe startup assumption. A physical limit switch should confirm this.
    "position":        "sheltered",
    "last_action":      None,
    "last_changed":     None,
    "vote_counts":      {"extend": 0, "retract": 0, "uncertain": 0},
    "last_prediction": None,
    "last_probability": None,
    "total_captures":   0,
    "pending_action":  None
}


def load_prediction_model():
    """Load the Keras model once; fail closed instead of guessing a command."""
    global model
    if model is None:
        with model_lock:
            if model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Model not found at {MODEL_PATH}. Set RACK_MODEL_PATH."
                    )
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print(f"[MODEL] Loaded: {MODEL_PATH}")
    return model

start_retention_worker()
start_discovery_service()

# ==========================================================
# ADAPTIVE CONFIG BASED ON TIME OF DAY
# ==========================================================
def get_adaptive_config(cfg: dict) -> dict:
    """
    Override ae_level and gain settings based on time of day.
    Prevents over-darkening at dusk/dawn and over-brightening at noon.
    User settings from the webpage are used as base,
    only ae_level and gainceiling are overridden adaptively.
    """
    result = cfg.copy()
    hour   = local_now().hour

    if 6 <= hour < 9:
        # Dawn — slightly boost exposure
        result["ae_level"]    = 1
        result["gainceiling"] = 5
        result["brightness"]  = 0
        print(f"[Adaptive] Dawn preset (hour={hour})")

    elif 9 <= hour < 16:
        # Bright midday — reduce exposure slightly
        result["ae_level"]    = -1
        result["gainceiling"] = 2
        result["brightness"]  = -1
        print(f"[Adaptive] Midday preset (hour={hour})")

    elif 16 <= hour < 18:
        # Late afternoon — neutral
        result["ae_level"]    = 0
        result["gainceiling"] = 4
        result["brightness"]  = 0
        print(f"[Adaptive] Afternoon preset (hour={hour})")

    elif 18 <= hour < 20:
        # Dusk — boost significantly
        result["ae_level"]    = 2
        result["gainceiling"] = 6
        result["brightness"]  = 2
        result["wb_mode"]     = 2  # Cloudy to counteract red cast
        print(f"[Adaptive] Dusk preset (hour={hour})")

    else:
        # Night — maximum boost
        result["ae_level"]    = 2
        result["gainceiling"] = 6
        result["brightness"]  = 2
        print(f"[Adaptive] Night preset (hour={hour})")

    return result

# Image correction functions are provided by image_processing.py

# ==========================================================
# MODEL INFERENCE & VOTING SYSTEM
# ==========================================================
def predict_image(image: Image.Image):
    """Return (frame_label, P(retract)) using training-identical preprocessing."""
    image = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    with model_lock:
        probability = float(load_prediction_model().predict(batch, verbose=0)[0][0])

    if probability >= RETRACT_PROBABILITY_THRESHOLD:
        return "retract", probability
    if probability <= EXTEND_PROBABILITY_THRESHOLD:
        return "extend", probability
    return "uncertain", probability


def set_position(action: str, reason: str, bypass_hysteresis=False):
    """Create a command only when movement is needed and permitted."""
    target = "sheltered" if action == "retract" else "outside"
    if rack_state["position"] == target:
        rack_state["pending_action"] = "hold"
        return f"{reason} Already {target}."

    last_changed_epoch = rack_state.get("last_changed_epoch")
    if (
        not bypass_hysteresis
        and last_changed_epoch is not None
        and time.time() - last_changed_epoch < MIN_OPPOSITE_ACTION_INTERVAL_SEC
    ):
        rack_state["pending_action"] = "hold"
        return f"{reason} Opposite movement blocked by two-minute hysteresis."

    rack_state["position"] = target
    rack_state["last_action"] = action
    rack_state["last_changed"] = local_now().isoformat(timespec="seconds")
    rack_state["last_changed_epoch"] = time.time()
    rack_state["pending_action"] = action
    return reason


def make_rack_decision(
    prediction: str,
    probability: float,
) -> dict:
    """
    Apply the operating schedule and camera-model voting.
    The latest five valid predictions form an overlapping FIFO window. After
    the first five captures, every new prediction removes the oldest one and
    immediately produces a new decision.
    """
    global rack_state
    with state_lock:
        rack_state["last_prediction"] = prediction
        rack_state["last_probability"] = round(probability, 6)
        rack_state["total_captures"] += 1
        rack_state["pending_action"] = "hold"

        hour = local_now().hour
        if not OPERATING_START_HOUR <= hour < OPERATING_END_HOUR:
            prediction_history.clear()
            reason = set_position(
                "retract", "Outside 08:00–18:00 operating hours.", True
            )
            return _build_response(reason)

        prediction_history.append(prediction)
        counts = collections.Counter(prediction_history)
        rack_state["vote_counts"] = {
            "extend": counts.get("extend", 0),
            "retract": counts.get("retract", 0),
            "uncertain": counts.get("uncertain", 0),
        }

        if len(prediction_history) < VOTE_WINDOW:
            reason = f"Building decision window: {len(prediction_history)}/{VOTE_WINDOW}."
            return _build_response(reason)

        retract_votes = counts.get("retract", 0)
        extend_votes = counts.get("extend", 0)
        completed_window = list(prediction_history)

        if retract_votes >= RETRACT_THRESHOLD:
            reason = set_position(
                "retract", f"Camera window: {retract_votes}/5 confident retract votes."
            )
        elif extend_votes >= EXTEND_THRESHOLD:
            reason = set_position(
                "extend", f"Camera window: {extend_votes}/5 confident extend votes."
            )
        else:
            reason = "No safe majority; retaining the current rack position."

        response = _build_response(reason)
        response["completed_window"] = completed_window
        return response


def _build_response(reason: str) -> dict:
    return {
        "action":          rack_state["pending_action"],
        "reason":          reason,
        "position":        rack_state["position"],
        "vote_counts":     rack_state["vote_counts"],
        "window":          list(prediction_history),
        "window_size":     len(prediction_history),
        "last_prediction": rack_state["last_prediction"],
        "probability_retract": rack_state["last_probability"],
        "timestamp":       local_now().isoformat(timespec="seconds")
    }

# ==========================================================
# UPLOAD ROUTE — Sends Plain Text Command to ESP32-CAM
# ==========================================================
@app.route('/upload', methods=['POST'])
def upload():
    global processing_enabled
    with process_lock:
        enabled = processing_enabled
    if not enabled:
        return Response("hold", mimetype='text/plain')

    if 'image' not in request.files:
        return "No image payload found", 400

    expire_fifo_images()

    file      = request.files['image']
    timestamp = local_now().strftime("%Y%m%d_%H%M%S_%f")
    filename  = f"weather_{timestamp}.jpg"
    filepath  = os.path.join(SAVE_DIR, filename)
    file.save(filepath)
    print(f"\n[!] Saved raw image: {filepath}")

    prediction = "uncertain"
    probability = 0.5

    try:
        img = Image.open(filepath).convert("RGB")
        img = apply_user_crop(img)
        total_frac, blob_frac, is_flagged = analyze_overexposure(img)

        print(f"    Overexposure: total={total_frac:.1%} "
              f"blob={blob_frac:.1%} flagged={is_flagged}")

        # Exposure, colour-cast, and white-balance correction
        img_fixed = correct_image(img)

        # Save corrected image
        corrected_path = os.path.join(CORRECTED_DIR, filename)
        img_fixed.save(corrected_path, quality=90)
        print(f"[!] Saved corrected: {corrected_path}")

        # Save original to flagged folder if overexposed
        if is_flagged:
            flagged_path = os.path.join(FLAGGED_DIR, filename)
            img.save(flagged_path, quality=90)
            print(f"[!] Flagged original: {flagged_path}")

        # MobileNetV2 classification. Train and infer using the same type of
        # image; this version uses the corrected ESP32 image.
        prediction, probability = predict_image(img_fixed)
        print(
            f"[!] Prediction: {prediction}; "
            f"P(retract)={probability:.4f}"
        )

    except Exception as e:
        # A processing/model failure is uncertain, never an automatic extend.
        print(f"[!] Processing/inference failed; holding safely: {e}")

    decision = make_rack_decision(prediction, probability)
    
    # Extract the raw action string ("retract", "extend", or "hold")
    action_command = decision.get("action") or "hold"
    
    print(f"[!] Action Command for ESP-NOW: {action_command}")
    print(f"[!] Reason: {decision['reason']}")

    # Return plain-text action to ESP32-CAM for immediate ESP-NOW forwarding
    return Response(action_command, mimetype='text/plain')

# ==========================================================
# CONFIG API — Returns adaptive settings to ESP32
# ==========================================================
@app.route('/config', methods=['GET'])
def get_config():
    cfg          = load_config()
    adaptive_cfg = get_adaptive_config(cfg)
    return jsonify(adaptive_cfg)


@app.route('/config/update', methods=['POST'])
def update_config():
    cfg           = load_config()
    CHECKBOX_KEYS = ("exposure_ctrl", "gain_ctrl", "whitebal", "awb_gain")

    for key in DEFAULT_CONFIG.keys():
        if key in request.form:
            if key in CHECKBOX_KEYS:
                cfg[key] = 1 if request.form.get(key) == "on" else 0
            else:
                try:
                    cfg[key] = int(request.form.get(key))
                except (ValueError, TypeError):
                    pass
        elif key in CHECKBOX_KEYS:
            cfg[key] = 0

    save_config(cfg)
    print(f"[!] Config updated: {cfg}")
    return settings_page()

# ==========================================================
# RACK STATUS API
# ==========================================================
@app.route('/rack/status', methods=['GET'])
def rack_status():
    with process_lock:
        enabled = processing_enabled
    return jsonify({
        "processing_enabled": enabled,
        "position":        rack_state["position"],
        "pending_action":  rack_state["pending_action"],
        "last_action":      rack_state["last_action"],
        "last_prediction": rack_state["last_prediction"],
        "last_probability": rack_state["last_probability"],
        "vote_counts":      rack_state["vote_counts"],
        "last_changed":     rack_state["last_changed"],
        "total_captures":   rack_state["total_captures"],
        "window_progress": f"{len(prediction_history)}/{VOTE_WINDOW}",
        "history":         list(prediction_history),
        "timestamp":       local_now().isoformat(timespec="seconds"),
    })


@app.route('/health', methods=['GET'])
def health():
    """Deployment health check without forcing model inference."""
    with process_lock:
        enabled = processing_enabled
    return jsonify({
        "status": "ok",
        "processing_enabled": enabled,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": model is not None,
        "operating_hours": "08:00-18:00",
        "timezone": str(LOCAL_TIMEZONE),
        "image_retention_minutes": IMAGE_RETENTION_SEC // 60,
        "window_type": "overlapping_fifo",
        "lan_discovery": "UDP 4210",
    })


@app.route('/rack/override', methods=['POST'])
def rack_override():
    position = request.form.get('position', 'outside')
    if position not in ('outside', 'sheltered'):
        return jsonify({"error": "Invalid rack position."}), 400
    with state_lock:
        rack_state["position"]       = position
        rack_state["pending_action"] = "extend" if position == "outside" else "retract"
        rack_state["last_action"]    = "manual_override"
        rack_state["last_changed"]   = local_now().isoformat(timespec="seconds")
        rack_state["last_changed_epoch"] = time.time()
        prediction_history.clear()
        rack_state["vote_counts"] = {"extend": 0, "retract": 0, "uncertain": 0}
        print(f"[RACK] Manual override → {position}")
    return jsonify(rack_state)


@app.route('/process/toggle', methods=['POST'])
def toggle_processing():
    """Pause/resume the complete automatic image-processing pipeline."""
    global processing_enabled
    requested = request.form.get("state", "toggle")
    with process_lock:
        if requested == "pause":
            processing_enabled = False
        elif requested == "resume":
            processing_enabled = True
        else:
            processing_enabled = not processing_enabled
        enabled = processing_enabled
    if not enabled:
        with state_lock:
            prediction_history.clear()
            rack_state["vote_counts"] = {"extend": 0, "retract": 0, "uncertain": 0}
            rack_state["pending_action"] = "hold"
    return jsonify({"processing_enabled": enabled, "command": "hold"})


@app.route('/rack/emergency-retract', methods=['POST'])
def emergency_retract():
    """Immediate safety retraction; bypasses camera voting and hysteresis."""
    with state_lock:
        prediction_history.clear()
        reason = set_position(
            "retract", "Emergency/safety input requested immediate retraction.", True
        )
        return jsonify(_build_response(reason))


@app.route('/crop/request', methods=['POST'])
def request_device_test_capture():
    """Queue one test capture for the connected ESP32-CAM."""
    request_id = local_now().strftime("%Y%m%d%H%M%S%f")
    with device_lock:
        test_capture.update({
            "request_id": request_id,
            "status": "requested",
            "filename": None,
            "requested_at": local_now().isoformat(timespec="seconds"),
        })
    return jsonify({"request_id": request_id, "status": "requested"})


@app.route('/device/instructions', methods=['GET'])
def device_instructions():
    """Polled by the ESP32-CAM for an operator-requested test capture."""
    with device_lock:
        if test_capture["status"] == "requested":
            return jsonify({
                "action": "capture_test",
                "request_id": test_capture["request_id"],
            })
    return jsonify({"action": "none"})


@app.route('/crop/device-upload', methods=['POST'])
def device_crop_upload():
    request_id = request.form.get("request_id", "")
    if not request_id.isdigit() or len(request_id) > 24:
        return jsonify({"error": "Invalid request ID."}), 400
    if "image" not in request.files:
        return jsonify({"error": "No image payload found."}), 400
    with device_lock:
        if request_id != test_capture.get("request_id"):
            return jsonify({"error": "Capture request is no longer active."}), 409

    filename = f"device_test_{request_id}.jpg"
    path = os.path.join(CROP_TEST_DIR, filename)
    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
        if image.width < 400 or image.height < 400:
            return jsonify({"error": "Camera image must be at least 400 × 400."}), 400
        image.save(path, quality=92)
    except (OSError, ValueError) as exc:
        return jsonify({"error": f"Invalid camera image: {exc}"}), 400

    with device_lock:
        test_capture.update({"status": "ready", "filename": filename})
    expire_fifo_images()
    return jsonify({"status": "ready", "request_id": request_id})


@app.route('/crop/status/<request_id>', methods=['GET'])
def crop_capture_status(request_id):
    with device_lock:
        if request_id != test_capture.get("request_id"):
            return jsonify({"status": "unknown"}), 404
        response = {"status": test_capture["status"]}
        if test_capture["status"] == "ready":
            response["image_url"] = url_for(
                "crop_preview", filename=test_capture["filename"]
            )
            response["filename"] = test_capture["filename"]
        return jsonify(response)


@app.route('/crop', methods=['GET', 'POST'])
def crop_setup():
    message = ""
    error = ""
    preview_name = None
    if request.method == 'POST':
        try:
            source_filename = request.form.get("source_filename", "")
            if (not source_filename.startswith("device_test_") or
                    not source_filename.endswith(".jpg") or
                    os.path.basename(source_filename) != source_filename):
                raise ValueError("Capture a test image from the ESP32-CAM first.")
            image = Image.open(os.path.join(CROP_TEST_DIR, source_filename)).convert("RGB")
            rotation = float(request.form.get("rotation", 0))
            x = int(float(request.form.get("crop_x", 0)))
            y = int(float(request.form.get("crop_y", 0)))
            width = int(float(request.form.get("crop_width", 0)))
            height = int(float(request.form.get("crop_height", 0)))
            image = validate_crop(image, rotation, x, y, width, height)

            cfg = {
                "enabled": True,
                "rotation": round(rotation, 1),
                "x": x, "y": y, "width": width, "height": height,
                "source_width": image.width,
                "source_height": image.height,
            }
            save_crop_config(cfg)
            preview_name = f"crop_preview_{local_now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            image.crop((x, y, x + width, y + height)).save(
                os.path.join(CROP_TEST_DIR, preview_name), quality=92
            )
            expire_fifo_images()
            message = "Crop saved. Future camera images will use this region."
        except (ValueError, TypeError, OSError) as exc:
            error = str(exc)

    cfg = load_crop_config()
    return Response(_crop_page_html(cfg, message, error, preview_name), mimetype='text/html')


@app.route('/crop/disable', methods=['POST'])
def crop_disable():
    cfg = load_crop_config()
    cfg["enabled"] = False
    save_crop_config(cfg)
    return redirect(url_for('crop_setup'))


@app.route('/crop/preview/<filename>')
def crop_preview(filename):
    return send_from_directory(CROP_TEST_DIR, filename)


def _crop_page_html(cfg, message="", error="", preview_name=None):
    status = "Enabled" if cfg.get("enabled") else "Disabled"
    preview = (f'<h3>Saved crop preview</h3><img class="preview" '
               f'src="/crop/preview/{preview_name}">') if preview_name else ""
    return f"""<!DOCTYPE html><html><head>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Crop Calibration</title>
    <style>
      *{{box-sizing:border-box}} body{{font-family:sans-serif;max-width:850px;
      margin:auto;padding:16px;background:#111;color:#eee}} a{{color:#8cf}}
      canvas{{max-width:100%;height:auto;border:2px solid #555;cursor:crosshair}}
      button,input[type=submit]{{padding:11px 16px;margin:6px;background:#287;
      color:white;border:0;border-radius:6px;cursor:pointer}} .danger{{background:#a43}}
      .notice{{padding:10px;border-radius:6px;background:#183}} .error{{background:#511}}
      .preview{{max-width:100%;border:2px solid #287}} .help{{color:#aaa}}
    </style></head><body>
    <p><a href="/">← Control panel</a></p>
    <h2>Rectangular Crop Calibration</h2>
    <p>Current crop: <strong>{status}</strong>. Upload a test image, rotate it if
    needed, then drag a rectangle that excludes the unwanted foreground. Both
    the test image and selected crop must be at least 400 × 400 pixels.</p>
    {f'<p class="notice">{message}</p>' if message else ''}
    {f'<p class="notice error">{error}</p>' if error else ''}
    <form id="cropForm" method="POST" enctype="multipart/form-data">
      <input id="testImage" type="file" name="test_image" accept="image/*" required>
      <div><button type="button" onclick="rotateBy(90)">Rotate 90° clockwise</button>
      <button type="button" onclick="clearSelection()">Clear selection</button></div>
      <canvas id="canvas"></canvas>
      <p id="selection" class="help">No crop selected.</p>
      <input type="hidden" id="rotation" name="rotation" value="0">
      <input type="hidden" id="cropX" name="crop_x"><input type="hidden" id="cropY" name="crop_y">
      <input type="hidden" id="cropW" name="crop_width"><input type="hidden" id="cropH" name="crop_height">
      <div><input type="submit" value="Save and apply crop"></div>
    </form>
    <form method="POST" action="/crop/disable">
      <button class="danger" type="submit">Disable cropping</button>
    </form>{preview}
    <script>
      const input=document.getElementById('testImage'), canvas=document.getElementById('canvas');
      const ctx=canvas.getContext('2d'); let img=new Image(), angle=0, start=null, box=null;
      input.onchange=()=>{{ const f=input.files[0]; if(!f)return;
        img.onload=()=>{{angle=0;document.getElementById('rotation').value=0;box=null;draw();}};
        img.src=URL.createObjectURL(f); }};
      function draw(){{ if(!img.width)return; const swap=angle%180!==0;
        canvas.width=swap?img.height:img.width; canvas.height=swap?img.width:img.height;
        ctx.save();ctx.translate(canvas.width/2,canvas.height/2);ctx.rotate(angle*Math.PI/180);
        ctx.drawImage(img,-img.width/2,-img.height/2);ctx.restore();
        if(box){{ctx.fillStyle='rgba(0,190,120,.22)';ctx.fillRect(box.x,box.y,box.w,box.h);
          ctx.strokeStyle='#00ef9a';ctx.lineWidth=Math.max(2,canvas.width/400);ctx.strokeRect(box.x,box.y,box.w,box.h);}}
        updateFields(); }}
      function point(e){{const r=canvas.getBoundingClientRect();return {{x:Math.round((e.clientX-r.left)*canvas.width/r.width),y:Math.round((e.clientY-r.top)*canvas.height/r.height)}};}}
      canvas.onmousedown=e=>{{start=point(e);box=null;}};
      canvas.onmousemove=e=>{{if(!start)return;const p=point(e);box={{x:Math.min(start.x,p.x),y:Math.min(start.y,p.y),w:Math.abs(p.x-start.x),h:Math.abs(p.y-start.y)}};draw();}};
      window.onmouseup=()=>{{start=null;updateFields();}};
      function rotateBy(v){{if(!img.width)return;angle=(angle+v)%360;box=null;document.getElementById('rotation').value=angle;draw();}}
      function clearSelection(){{box=null;draw();}}
      function updateFields(){{const t=document.getElementById('selection');
        if(!box){{['cropX','cropY','cropW','cropH'].forEach(id=>document.getElementById(id).value='');t.textContent='No crop selected.';return;}}
        document.getElementById('cropX').value=box.x;document.getElementById('cropY').value=box.y;
        document.getElementById('cropW').value=box.w;document.getElementById('cropH').value=box.h;
        t.textContent=`Crop: ${{box.w}} × ${{box.h}} pixels at (${{box.x}}, ${{box.y}})`;}}
      document.getElementById('cropForm').onsubmit=e=>{{if(!box||box.w<400||box.h<400){{e.preventDefault();alert('Select a crop of at least 400 × 400 pixels.');}}}};
    </script></body></html>"""


# The refreshed UI is kept separately so server routes remain readable.
from web_ui import crop_page_html as _crop_page_html

# ==========================================================
# IMAGE ROUTES
# ==========================================================
@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(CORRECTED_DIR, filename)

@app.route('/raw/<filename>')
def get_raw_image(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.route('/flagged/<filename>')
def get_flagged_image(filename):
    return send_from_directory(FLAGGED_DIR, filename)

@app.route('/latest')
def latest_image():
    images = sorted(os.listdir(CORRECTED_DIR), reverse=True)
    if images:
        return send_from_directory(CORRECTED_DIR, images[0])
    return "No images yet", 404

@app.route('/latest/raw')
def latest_raw():
    images = sorted(os.listdir(SAVE_DIR), reverse=True)
    if images:
        return send_from_directory(SAVE_DIR, images[0])
    return "No images yet", 404

# ==========================================================
# GALLERY
# ==========================================================
@app.route('/gallery')
def gallery():
    corrected = sorted(os.listdir(CORRECTED_DIR), reverse=True)
    flagged   = sorted(os.listdir(FLAGGED_DIR),   reverse=True)

    def img_card(img, route, border_color, label):
        return (f'<div style="margin:8px;display:inline-block;'
                f'vertical-align:top;text-align:center">'
                f'<p style="color:#aaa;font-size:0.7em;margin:3px 0">'
                f'{img}</p>'
                f'<img src="/{route}/{img}" width="280" '
                f'style="border-radius:8px;border:2px solid {border_color}">'
                f'<p style="color:{border_color};font-size:0.75em">'
                f'{label}</p></div>')

    corrected_tags = "".join([
        img_card(i, "image", "#2a7", "✅ Corrected")
        for i in corrected if i.endswith('.jpg')
    ])
    flagged_tags = "".join([
        img_card(i, "flagged", "#a72", "⚠️ Overexposed")
        for i in flagged if i.endswith('.jpg')
    ])

    raw_count  = len(os.listdir(SAVE_DIR))
    corr_count = len(corrected)
    flag_count = len(flagged)

    votes = rack_state["vote_counts"]

    return f"""<!DOCTYPE html><html>
    <head>
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Gallery</title>
        <style>
            body{{background:#f3f6f9;color:#17212b;max-width:1100px;margin:auto;
                  font-family:Inter,Segoe UI,sans-serif;padding:24px}}
            h1{{color:#176b87}} h2{{color:#176b87;margin-top:24px}}
            a{{color:#176b87;text-decoration:none}}
            .stats{{background:white;border-radius:12px;box-shadow:0 4px 16px #19324712;
                    padding:16px;margin:12px 0;font-size:14px}}
            .nav{{display:flex;gap:8px;margin-bottom:16px}}
            .nav a{{flex:1;text-align:center;padding:10px;
                    background:white;border-radius:8px;font-size:13px}}
        </style>
    </head>
    <body>
        <h1>🖼️ Image Gallery</h1>
        <div class="nav">
            <a href="/">⚙️ Settings</a>
            <a href="/gallery">🖼️ Gallery</a>
            <a href="/latest">📸 Latest</a>
            <a href="/rack/status">🏠 Rack</a>
        </div>
        <div class="stats">
            📸 Raw: {raw_count} &nbsp;|&nbsp;
            ✅ Corrected: {corr_count} &nbsp;|&nbsp;
            ⚠️ Flagged: {flag_count}<br>
            🏠 Rack: <strong>{rack_state['position']}</strong>
            &nbsp;|&nbsp;
            Window: Extend {votes['extend']} | Retract {votes['retract']} |
            Uncertain {votes['uncertain']}
            ({len(prediction_history)}/{VOTE_WINDOW})
        </div>
        <h2>✅ Corrected Images</h2>
        <div>{corrected_tags or
              "<p style='color:#666'>No corrected images yet.</p>"}</div>
        <h2>⚠️ Flagged Overexposed</h2>
        <div>{flagged_tags or
              "<p style='color:#666'>No flagged images yet.</p>"}</div>
    </body></html>"""

# ==========================================================
# SETTINGS PAGE
# ==========================================================
@app.route('/', methods=['GET'])
def settings_page():
    cfg  = load_config()
    crop_cfg = load_crop_config()
    with process_lock:
        enabled = processing_enabled
    hour = local_now().hour
    votes = rack_state["vote_counts"]

    def checked(key):
        return "checked" if cfg.get(key) == 1 else ""

    effect_options = ""
    for val, label in [(0,"Normal (Color)"), (2,"Grayscale (B&W)"),
                       (1,"Negative"), (6,"Sepia")]:
        sel = "selected" if cfg.get("special_effect", 0) == val else ""
        effect_options += f'<option value="{val}" {sel}>{label}</option>\n'

    wb_options = ""
    for val, label in [(0,"Auto"),(1,"Sunny"),(2,"Cloudy"),
                       (3,"Office"),(4,"Home")]:
        sel = "selected" if cfg.get("wb_mode", 0) == val else ""
        wb_options += f'<option value="{val}" {sel}>{label}</option>\n'

    raw_count  = len(os.listdir(SAVE_DIR))
    corr_count = len(os.listdir(CORRECTED_DIR))
    flag_count = len(os.listdir(FLAGGED_DIR))

    rack_color = "#2a7" if rack_state["position"] == "outside" else "#a72"
    rack_label = ("☀️ Outside" if rack_state["position"] == "outside"
                  else "🏠 Sheltered")

    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>ESP32-CAM Control Panel</title>
        <style>
            *{{box-sizing:border-box}}
            body{{font-family:Inter,Segoe UI,sans-serif;max-width:720px;margin:0 auto;
                  padding:24px;background:#f3f6f9;color:#17212b}}
            h2{{text-align:center;color:#176b87;margin-bottom:14px}}
            h3{{color:#176b87;border-bottom:1px solid #dbe3e8;
                padding-bottom:6px;margin-top:24px}}
            .row{{margin-bottom:18px}}
            label{{display:block;font-weight:bold;margin-bottom:4px}}
            .val{{float:right;font-weight:normal;color:#176b87}}
            input[type=range]{{width:100%;accent-color:#176b87;margin-top:4px}}
            input[type=checkbox]{{transform:scale(1.4);margin-right:8px;
                                  accent-color:#176b87}}
            select{{width:100%;padding:10px;font-size:15px;
                    border-radius:8px;background:white;color:#17212b;
                    border:1px solid #cbd5dc;margin-top:4px}}
            button{{width:100%;padding:14px;font-size:16px;
                    background:#176b87;color:white;border:none;
                    border-radius:8px;margin-top:16px;cursor:pointer}}
            button:hover{{filter:brightness(.94)}}
            .hint{{font-size:12px;color:#64748b;margin-top:3px}}
            .nav{{display:flex;gap:8px;margin-bottom:16px}}
            .nav a{{flex:1;text-align:center;padding:10px;background:white;
                    color:#176b87;border-radius:8px;text-decoration:none;
                    font-size:13px}}
            .nav a:hover{{background:#e8f2f5}}
            .status{{background:white;border-radius:12px;padding:14px;
                     text-align:center;margin-bottom:16px;font-size:13px;
                     color:#334155;box-shadow:0 4px 16px #19324712}}
            .rack-status{{background:white;border-radius:12px;padding:16px;
                          margin-bottom:12px;box-shadow:0 4px 16px #19324712}}
            .preview{{width:100%;border-radius:8px;margin-top:8px;
                      border:1px solid #dbe3e8}}
            .override-btns{{display:flex;gap:10px;margin-top:8px}}
            .override-btns button{{margin-top:0}}
            .system-btns{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}}
            .system-btns button{{margin:0}} .start{{background:#18864b}} .stop{{background:#b33a3a}}
            hr{{border:none;border-top:1px solid #dbe3e8;margin:8px 0}}
            .time-badge{{background:#edf3f6;border-radius:4px;padding:2px 8px;
                         font-size:12px;color:#176b87;float:right}}
        </style>
    </head>
    <body>
        <h2>📷 ESP32-CAM Control Panel</h2>

        <div class="nav">
            <a href="/">⚙️ Settings</a>
            <a href="/gallery">🖼️ Gallery</a>
            <a href="/latest">📸 Latest</a>
            <a href="/rack/status">🏠 Rack</a>
            <a href="/crop">✂️ Crop</a>
        </div>

        <div class="status">
            <span id="systemState">{'🟢 Running' if enabled else '🔴 Process paused'}</span> &nbsp;|&nbsp;
            📸 {raw_count} captured &nbsp;|&nbsp;
            ✅ {corr_count} corrected &nbsp;|&nbsp;
            ⚠️ {flag_count} flagged<br>
            ✂️ Crop: {'enabled' if crop_cfg.get('enabled') else 'disabled'}
            &nbsp;|&nbsp; Retention: 10 minutes
        </div>

        <div class="system-btns">
            <button class="start" type="button" onclick="setProcess('resume')">▶ Start system</button>
            <button class="stop" type="button" onclick="setProcess('pause')">■ Interrupt system</button>
        </div>

        <!-- RACK STATUS -->
        <div class="rack-status">
            <strong>🏠 Rack Status</strong>
            <span id="rackTime" class="time-badge">
                {local_now().strftime('%H:%M')}
            </span><br><br>
            <span id="rackPosition" style="color:{rack_color};font-size:1.1em">
                {rack_label}
            </span>
            &nbsp;|&nbsp;
            Last: <span id="lastPrediction">{rack_state.get('last_prediction', 'None')}</span><br>
            <small id="voteStatus" style="color:#666">
                Window: Extend {votes['extend']} | Retract {votes['retract']}
                | Uncertain {votes['uncertain']} ({len(prediction_history)}/{VOTE_WINDOW})
                &nbsp;|&nbsp; History: {list(prediction_history) or 'empty'}
            </small>
        </div>

        <!-- MANUAL RACK OVERRIDE -->
        <div>
            <div class="override-btns">
                <button type="button" onclick="manualOverride('outside')"
                        style="background:#2a7">
                    ☀️ Force Outside
                </button>
                <button type="button" onclick="manualOverride('sheltered')"
                        style="background:#a72">
                    🏠 Force Sheltered
                </button>
            </div>
        </div>

        <!-- LATEST IMAGE -->
        <div class="row" style="margin-top:16px">
            <label>Latest Corrected Image</label>
            {"<img src='/latest' class='preview'>"
             if corr_count > 0
             else "<p class='hint'>No images captured yet.</p>"}
        </div>

        <!-- ADAPTIVE TIME NOTICE -->
        <div style="background:#e8f2f5;border-radius:8px;padding:10px 12px;
                    margin-bottom:8px;font-size:12px;color:#176b87">
            ⏰ Current time: {hour:02d}:00 —
            Adaptive preset auto-overrides ae_level & gainceiling.
            Your manual settings are saved but may be adjusted.
        </div>

        <form method="POST" action="/config/update">

            <!-- BRIGHTNESS & CONTRAST -->
            <h3>☀️ Brightness & Contrast</h3>

            <div class="row">
                <label>Brightness
                    <span class="val" id="brightness_val">
                        {cfg.get('brightness', 0)}
                    </span>
                </label>
                <input type="range" name="brightness"
                       min="-2" max="2" value="{cfg.get('brightness', 0)}"
                       oninput="document.getElementById(
                           'brightness_val').innerText=this.value">
                <div class="hint">
                    Adaptive preset overrides this at dawn/dusk/night.
                </div>
            </div>

            <div class="row">
                <label>Contrast
                    <span class="val" id="contrast_val">
                        {cfg.get('contrast', 1)}
                    </span>
                </label>
                <input type="range" name="contrast"
                       min="-2" max="2" value="{cfg.get('contrast', 1)}"
                       oninput="document.getElementById(
                           'contrast_val').innerText=this.value">
            </div>

            <div class="row">
                <label>Saturation
                    <span class="val" id="saturation_val">
                        {cfg.get('saturation', 0)}
                    </span>
                </label>
                <input type="range" name="saturation"
                       min="-2" max="2" value="{cfg.get('saturation', 0)}"
                       oninput="document.getElementById(
                           'saturation_val').innerText=this.value">
                <div class="hint">
                    -1 to -2 reduces color cast from OV2640 sensor.
                </div>
            </div>

            <!-- EXPOSURE -->
            <h3>📡 Exposure Control</h3>

            <div class="row">
                <label>
                    <input type="checkbox" name="exposure_ctrl"
                           {checked('exposure_ctrl')}
                           onchange="toggle('aec_row', !this.checked)">
                    Auto Exposure (AEC)
                </label>
                <div class="hint">Keep on for automatic adjustment.</div>
            </div>

            <div class="row" id="aec_row">
                <label>Manual Exposure Value
                    <span class="val" id="aec_value_val">
                        {cfg.get('aec_value', 400)}
                    </span>
                </label>
                <input type="range" name="aec_value"
                       min="0" max="1200"
                       value="{cfg.get('aec_value', 400)}"
                       oninput="document.getElementById(
                           'aec_value_val').innerText=this.value">
                <div class="hint">Active only when AEC is off.</div>
            </div>

            <div class="row">
                <label>AE Level (Exposure Compensation)
                    <span class="val" id="ae_level_val">
                        {cfg.get('ae_level', 0)}
                    </span>
                </label>
                <input type="range" name="ae_level"
                       min="-2" max="2" value="{cfg.get('ae_level', 0)}"
                       oninput="document.getElementById(
                           'ae_level_val').innerText=this.value">
                <div class="hint">
                    0 = neutral. Adaptive preset overrides this
                    based on time of day.
                </div>
            </div>

            <!-- GAIN -->
            <h3>📶 Gain Control</h3>

            <div class="row">
                <label>
                    <input type="checkbox" name="gain_ctrl"
                           {checked('gain_ctrl')}
                           onchange="toggle('agc_row', !this.checked)">
                    Auto Gain (AGC)
                </label>
                <div class="hint">Keep on for automatic adjustment.</div>
            </div>

            <div class="row" id="agc_row">
                <label>Manual Gain
                    <span class="val" id="agc_gain_val">
                        {cfg.get('agc_gain', 0)}
                    </span>
                </label>
                <input type="range" name="agc_gain"
                       min="0" max="30"
                       value="{cfg.get('agc_gain', 0)}"
                       oninput="document.getElementById(
                           'agc_gain_val').innerText=this.value">
                <div class="hint">Active only when AGC is off.</div>
            </div>

            <div class="row">
                <label>Gain Ceiling
                    <span class="val" id="gainceiling_val">
                        {cfg.get('gainceiling', 4)}
                    </span>
                </label>
                <input type="range" name="gainceiling"
                       min="0" max="6"
                       value="{cfg.get('gainceiling', 4)}"
                       oninput="document.getElementById(
                           'gainceiling_val').innerText=this.value">
                <div class="hint">
                    Higher = more gain in low light. Adaptive preset
                    overrides based on time.
                </div>
            </div>

            <!-- IMAGE MODE -->
            <h3>🎨 Image Mode</h3>

            <div class="row">
                <label>Special Effect</label>
                <select name="special_effect">
                    {effect_options}
                </select>
                <div class="hint">
                    Grayscale removes color cast — good for ML training.
                </div>
            </div>

            <!-- WHITE BALANCE -->
            <h3>💡 White Balance</h3>

            <div class="row">
                <label>
                    <input type="checkbox" name="whitebal"
                           {checked('whitebal')}>
                    Auto White Balance (AWB)
                </label>
                <div class="hint">Corrects color cast from OV2640.</div>
            </div>

            <div class="row">
                <label>
                    <input type="checkbox" name="awb_gain"
                           {checked('awb_gain')}>
                    AWB Gain
                </label>
                <div class="hint">Keep on alongside AWB.</div>
            </div> 

            <div class="row">
                <label>White Balance Mode</label>
                <select name="wb_mode">
                    {wb_options}
                </select>
                <div class="hint">
                    Auto = best for general use.
                    Cloudy = cooler tone, reduces red/warm cast at dusk.
                </div>
            </div>

            <hr>
            <button type="submit">✅ Save Settings</button>

        </form>

        <p class="hint" style="text-align:center;margin-top:12px">
            Changes apply to the next scheduled ESP32-CAM capture.
        </p>

        <script>
            async function manualOverride(position) {{
                const response = await fetch('/rack/override', {{
                    method:'POST',
                    headers:{{'Content-Type':'application/x-www-form-urlencoded'}},
                    body:'position=' + encodeURIComponent(position)
                }});
                if (!response.ok) {{
                    alert('Unable to update the rack position.');
                    return;
                }}
                await refreshStatus();
            }}
            async function setProcess(state) {{
                await fetch('/process/toggle', {{
                    method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}},
                    body:'state=' + state
                }});
                await refreshStatus();
            }}
            async function refreshStatus() {{
                try {{
                    const response = await fetch('/rack/status', {{cache:'no-store'}});
                    if (!response.ok) return;
                    const data = await response.json();
                    const position = document.getElementById('rackPosition');
                    position.textContent = data.position === 'outside' ? '☀️ Outside' : '🏠 Sheltered';
                    position.style.color = data.position === 'outside' ? '#18864b' : '#a66a21';
                    document.getElementById('lastPrediction').textContent = data.last_prediction || 'None';
                    document.getElementById('systemState').textContent =
                        data.processing_enabled ? '🟢 Running' : '🔴 Process paused';
                    const votes = data.vote_counts || {{extend:0,retract:0,uncertain:0}};
                    const history = data.history && data.history.length ?
                        JSON.stringify(data.history) : 'empty';
                    document.getElementById('voteStatus').textContent =
                        `Window: Extend ${{votes.extend}} | Retract ${{votes.retract}} | ` +
                        `Uncertain ${{votes.uncertain}} (${{data.window_progress}}) | History: ${{history}}`;
                    if (data.timestamp) {{
                        document.getElementById('rackTime').textContent = data.timestamp.slice(11,16);
                    }}
                }} catch (error) {{
                    document.getElementById('systemState').textContent = '⚠️ Connection lost';
                }}
            }}
            function toggle(id, disabled) {{
                const el = document.getElementById(id);
                if (!el) return;
                el.style.opacity = disabled ? '0.4' : '1';
                el.querySelectorAll('input').forEach(
                    i => i.disabled = disabled);
            }}
            window.onload = function() {{
                toggle('aec_row',
                    {str(cfg.get('exposure_ctrl', 1) == 1).lower()});
                toggle('agc_row',
                    {str(cfg.get('gain_ctrl', 1) == 1).lower()});
                refreshStatus();
                setInterval(refreshStatus, 2000);
            }};
        </script>
    </body>
    </html>"""
    return Response(html, mimetype='text/html')

# ==========================================================
# MAIN
# ==========================================================
if __name__ == '__main__':
    # Use one process because voting state is held in memory. If deploying with
    # multiple workers, move rack_state/history to Redis or a database.
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get("PORT", "8080")),
        debug=False,
        threaded=True,
    )
