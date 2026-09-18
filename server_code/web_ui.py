def crop_page_html(config, message="", error="", preview_name=None):
    status = "Enabled" if config.get("enabled") else "Disabled"
    return f"""<!doctype html><html><head>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Camera crop setup</title><style>
    *{{box-sizing:border-box}}body{{font-family:Inter,Segoe UI,sans-serif;max-width:960px;
    margin:auto;padding:24px;background:#f3f6f9;color:#17212b}}a{{color:#176b87;text-decoration:none}}
    .card{{background:#fff;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 4px 18px #19324714}}
    canvas{{display:block;max-width:100%;height:auto;margin:auto;border-radius:10px;background:#e7edf2;cursor:crosshair}}
    button,input[type=submit]{{padding:11px 15px;margin:4px;border:0;border-radius:8px;background:#176b87;
    color:#fff;font-weight:600;cursor:pointer}}button:disabled,input:disabled{{opacity:.45;cursor:not-allowed}}
    .secondary{{background:#64748b}}.danger{{background:#b33a3a}}.toolbar{{display:flex;flex-wrap:wrap;
    gap:6px;align-items:center;margin:12px 0}}.notice{{padding:11px;border-radius:8px;background:#e4f5ec;color:#17633a}}
    .error{{background:#fde8e8;color:#8c2424}}.help{{color:#64748b}}input[type=range]{{width:210px}}
    </style></head><body><p><a href="/">← Control panel</a></p><h1>Camera crop setup</h1>
    <p class="help">Current crop: <strong>{status}</strong>. Capture a live camera image, rotate it,
    then drag a rectangular region containing clear sky and no foreground objects.</p>
    {f'<p class="notice">{message}</p>' if message else ''}
    {f'<p class="notice error">{error}</p>' if error else ''}
    <div class="card"><h2>1. Capture from ESP32-CAM</h2>
      <button id="captureButton" onclick="requestCapture()">Capture test image</button>
      <span id="captureStatus" class="help">Camera must be powered on and connected.</span></div>
    <form id="cropForm" method="post" class="card"><h2>2. Rotate and select</h2>
      <div class="toolbar"><button type="button" onclick="rotateBy(-90)">↶ Left 90°</button>
      <button type="button" onclick="rotateBy(90)">↷ Right 90°</button>
      <button type="button" onclick="rotateBy(180)">Rotate 180°</button>
      <button type="button" class="secondary" onclick="setAngle(0)">Reset rotation</button>
      <label>Fine angle <input id="angleSlider" type="range" min="-180" max="180" step="1" value="0"
      oninput="setAngle(Number(this.value))"> <span id="angleText">0°</span></label></div>
      <canvas id="canvas"></canvas><p id="selection" class="help">Waiting for camera image.</p>
      <input type="hidden" id="sourceFilename" name="source_filename"><input type="hidden" id="rotation" name="rotation" value="0">
      <input type="hidden" id="cropX" name="crop_x"><input type="hidden" id="cropY" name="crop_y">
      <input type="hidden" id="cropW" name="crop_width"><input type="hidden" id="cropH" name="crop_height">
      <input id="saveCrop" type="submit" value="Save and apply crop" disabled></form>
    <form method="post" action="/crop/disable" class="card"><h2>3. Crop control</h2>
      <button class="danger" type="submit">Disable cropping</button></form>
    <script>
    const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
    let img=new Image(),angle=0,start=null,box=null;
    async function requestCapture(){{const b=document.getElementById('captureButton'),s=document.getElementById('captureStatus');
      b.disabled=true;s.textContent='Waiting for ESP32-CAM…';try{{const r=await fetch('/crop/request',{{method:'POST'}});
      const d=await r.json();await pollCapture(d.request_id);}}catch(e){{s.textContent='Request failed.';b.disabled=false;}}}}
    async function pollCapture(id){{const s=document.getElementById('captureStatus'),b=document.getElementById('captureButton');
      for(let n=0;n<60;n++){{await new Promise(r=>setTimeout(r,1000));const r=await fetch('/crop/status/'+id);
      if(!r.ok)continue;const d=await r.json();if(d.status==='ready'){{document.getElementById('sourceFilename').value=d.filename;
      img.onload=()=>{{setAngle(0);s.textContent='Live image received. Select the sky region.';b.disabled=false;}};
      img.src=d.image_url+'?t='+Date.now();return;}}}}s.textContent='Timed out. Check the camera and try again.';b.disabled=false;}}
    function draw(){{if(!img.width)return;const r=angle*Math.PI/180,c=Math.abs(Math.cos(r)),s=Math.abs(Math.sin(r));
      canvas.width=Math.ceil(img.width*c+img.height*s);canvas.height=Math.ceil(img.width*s+img.height*c);
      ctx.save();ctx.translate(canvas.width/2,canvas.height/2);ctx.rotate(r);ctx.drawImage(img,-img.width/2,-img.height/2);ctx.restore();
      if(box){{ctx.fillStyle='rgba(0,175,115,.22)';ctx.fillRect(box.x,box.y,box.w,box.h);ctx.strokeStyle='#00a873';
      ctx.lineWidth=Math.max(2,canvas.width/400);ctx.strokeRect(box.x,box.y,box.w,box.h);}}updateFields();}}
    function point(e){{const r=canvas.getBoundingClientRect();return{{x:Math.round((e.clientX-r.left)*canvas.width/r.width),
      y:Math.round((e.clientY-r.top)*canvas.height/r.height)}}}}canvas.onmousedown=e=>{{if(img.width){{start=point(e);box=null;}}}};
    canvas.onmousemove=e=>{{if(!start)return;const p=point(e);box={{x:Math.min(start.x,p.x),y:Math.min(start.y,p.y),
      w:Math.abs(p.x-start.x),h:Math.abs(p.y-start.y)}};draw();}};window.onmouseup=()=>{{start=null;updateFields();}};
    function rotateBy(v){{let n=angle+v;while(n>180)n-=360;while(n<-180)n+=360;setAngle(n)}}
    function setAngle(v){{if(!img.width)return;angle=Math.max(-180,Math.min(180,v));box=null;
      document.getElementById('rotation').value=angle;document.getElementById('angleSlider').value=angle;
      document.getElementById('angleText').textContent=angle+'°';draw();}}
    function updateFields(){{const t=document.getElementById('selection'),save=document.getElementById('saveCrop');
      if(!box){{['cropX','cropY','cropW','cropH'].forEach(id=>document.getElementById(id).value='');
      t.textContent=img.width?'Drag a rectangle over clear sky.':'Waiting for camera image.';save.disabled=true;return;}}
      cropX.value=box.x;cropY.value=box.y;cropW.value=box.w;cropH.value=box.h;const valid=box.w>=400&&box.h>=400;
      save.disabled=!valid;t.textContent=`Crop: ${{box.w}} × ${{box.h}} pixels ${{valid?'✓':'— minimum 400 × 400'}}`;}}
    cropForm.onsubmit=e=>{{if(!box||box.w<400||box.h<400)e.preventDefault();}};
    </script></body></html>"""
