from PIL import Image

from app_config import load_crop_config


def rotate_clockwise(image: Image.Image, rotation: int) -> Image.Image:
    rotation %= 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")
    return image.rotate(-rotation, expand=True) if rotation else image


def apply_user_crop(image: Image.Image) -> Image.Image:
    """Apply the calibrated rectangular crop proportionally to a new image."""
    cfg = load_crop_config()
    if not cfg.get("enabled"):
        return image

    image = rotate_clockwise(image, int(cfg.get("rotation", 0)))
    source_w = int(cfg.get("source_width", 0))
    source_h = int(cfg.get("source_height", 0))
    if source_w <= 0 or source_h <= 0:
        raise ValueError("Invalid crop calibration dimensions.")

    scale_x = image.width / source_w
    scale_y = image.height / source_h
    x = max(0, round(int(cfg["x"]) * scale_x))
    y = max(0, round(int(cfg["y"]) * scale_y))
    right = min(image.width, x + round(int(cfg["width"]) * scale_x))
    bottom = min(image.height, y + round(int(cfg["height"]) * scale_y))
    if right <= x or bottom <= y:
        raise ValueError("Configured crop lies outside the uploaded image.")
    return image.crop((x, y, right, bottom))


def validate_crop(image, rotation, x, y, width, height):
    if image.width < 400 or image.height < 400:
        raise ValueError("The test image must be at least 400 × 400 pixels.")
    image = rotate_clockwise(image, rotation)
    if width < 400 or height < 400:
        raise ValueError("The crop rectangle must be at least 400 × 400 pixels.")
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        raise ValueError("The crop rectangle must remain inside the rotated image.")
    return image
