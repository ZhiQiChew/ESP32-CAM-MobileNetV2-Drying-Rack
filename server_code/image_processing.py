import numpy as np
from PIL import Image
from scipy import ndimage

from app_config import (
    CORONA_BLOB_FRACTION_THRESHOLD, OVEREXPOSED_FRACTION_THRESHOLD,
    OVEREXPOSED_PIXEL_THRESHOLD,
)


def analyze_overexposure(image: Image.Image):
    array = np.asarray(image)
    saturated = np.all(array >= OVEREXPOSED_PIXEL_THRESHOLD, axis=2)
    total_fraction = saturated.mean()
    largest_blob_fraction = 0.0
    if saturated.any():
        labeled, count = ndimage.label(saturated)
        if count > 0:
            sizes = ndimage.sum(saturated, labeled, range(1, count + 1))
            largest_blob_fraction = sizes.max() / saturated.size
    flagged = (
        total_fraction >= OVEREXPOSED_FRACTION_THRESHOLD
        or largest_blob_fraction >= CORONA_BLOB_FRACTION_THRESHOLD
    )
    return total_fraction, largest_blob_fraction, flagged


def fix_exposure(image: Image.Image) -> Image.Image:
    array = np.asarray(image).astype(np.float32) / 255.0
    brightness = array.mean()
    if brightness > 0.65:
        gamma = max(0.45, 1.0 - (brightness - 0.65) * 2.0)
        array = np.power(np.clip(array, 1e-6, 1.0), 1.0 / gamma)
    elif brightness < 0.30:
        boost = min(0.30 / max(brightness, 0.05), 3.0)
        array = np.power(np.clip(array, 1e-6, 1.0), 1.0 / boost)
    else:
        return image
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def fix_color_cast(image: Image.Image) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    usable = np.all(array < 200, axis=2)
    if usable.sum() < 500:
        return image
    r_mean, g_mean, b_mean = [array[:, :, channel][usable].mean() for channel in range(3)]
    r_dominance, b_dominance = r_mean - g_mean, b_mean - g_mean
    if r_dominance > 15 and b_dominance < 5:
        array[:, :, 0] *= min(g_mean / r_mean, 1.0)
        if b_mean < g_mean:
            array[:, :, 2] *= min(1.15, (g_mean / b_mean) * 1.1)
    elif r_dominance > 5 and b_dominance > 5:
        if r_mean > g_mean:
            array[:, :, 0] *= min((g_mean / r_mean) * 1.05, 1.0)
        if b_mean > g_mean:
            array[:, :, 2] *= min((g_mean / b_mean) * 0.95, 1.0)
            array[:, :, 1] *= 1.05
    else:
        return image
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def gray_world_white_balance(image: Image.Image) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    usable = np.all(array < 200, axis=2)
    if usable.sum() < 1000:
        return image
    means = [array[:, :, channel][usable].mean() for channel in range(3)]
    gray = sum(means) / 3.0
    for channel, mean in enumerate(means):
        array[:, :, channel] *= min(gray / mean, 1.20)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def correct_image(image: Image.Image) -> Image.Image:
    image = fix_exposure(image)
    image = fix_color_cast(image)
    return gray_world_white_balance(image)
