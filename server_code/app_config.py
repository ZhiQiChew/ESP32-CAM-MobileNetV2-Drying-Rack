import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

SAVE_DIR = "captured_images"
CORRECTED_DIR = "captured_images_corrected"
FLAGGED_DIR = "captured_images_flagged_overexposed"
CROP_TEST_DIR = "crop_test_images"
CONFIG_FILE = "camera_config.json"
CROP_CONFIG_FILE = "crop_config.json"
SLEEP_DURATION_SEC = 30
IMAGE_RETENTION_SEC = 10 * 60

MODEL_PATH = os.environ.get(
    "RACK_MODEL_PATH", "exported_models/extend_retract_mobilenetv2.keras"
)
IMG_SIZE = (224, 224)
LOCAL_TIMEZONE = ZoneInfo(os.environ.get("RACK_TIMEZONE", "Asia/Kuala_Lumpur"))
OPERATING_START_HOUR = 8
OPERATING_END_HOUR = 18

RETRACT_PROBABILITY_THRESHOLD = 0.80
EXTEND_PROBABILITY_THRESHOLD = 0.20
VOTE_WINDOW = 5
RETRACT_THRESHOLD = 3
EXTEND_THRESHOLD = 3
MIN_OPPOSITE_ACTION_INTERVAL_SEC = 2 * 60

OVEREXPOSED_PIXEL_THRESHOLD = 240
OVEREXPOSED_FRACTION_THRESHOLD = 0.10
CORONA_BLOB_FRACTION_THRESHOLD = 0.04

DEFAULT_CONFIG = {
    "exposure_ctrl": 1, "aec_value": 400, "gain_ctrl": 1,
    "agc_gain": 0, "gainceiling": 4, "brightness": 0,
    "contrast": 1, "saturation": 0, "whitebal": 1,
    "awb_gain": 1, "wb_mode": 0, "special_effect": 0, "ae_level": 0,
}

DEFAULT_CROP_CONFIG = {
    "enabled": False, "rotation": 0, "x": 0, "y": 0,
    "width": 0, "height": 0, "source_width": 0, "source_height": 0,
}

for directory in (SAVE_DIR, CORRECTED_DIR, FLAGGED_DIR, CROP_TEST_DIR):
    os.makedirs(directory, exist_ok=True)


def local_now():
    return datetime.now(LOCAL_TIMEZONE)


def _load_json(path, defaults):
    if not os.path.exists(path):
        _save_json(path, defaults)
        return defaults.copy()
    try:
        with open(path, "r") as file:
            return {**defaults, **json.load(file)}
    except (OSError, json.JSONDecodeError, TypeError):
        return defaults.copy()


def _save_json(path, value):
    with open(path, "w") as file:
        json.dump(value, file, indent=2)


def load_config():
    return _load_json(CONFIG_FILE, DEFAULT_CONFIG)


def save_config(config):
    _save_json(CONFIG_FILE, config)


def load_crop_config():
    return _load_json(CROP_CONFIG_FILE, DEFAULT_CROP_CONFIG)


def save_crop_config(config):
    _save_json(CROP_CONFIG_FILE, config)
