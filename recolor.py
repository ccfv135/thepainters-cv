import numpy as np
from PIL import Image, ImageFilter

_D65 = (0.95047, 1.0, 1.08883)

FEATHER_RADIUS = 3
SPECULAR_PERCENTILE = 97
SPECULAR_CHROMA_FACTOR = 0.35


def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(out * 255.0, 0, 255)


def _f(t):
    delta = 6 / 29
    return np.where(t > delta**3, np.cbrt(t), t / (3 * delta**2) + 4 / 29)


def _finv(t):
    delta = 6 / 29
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)

    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    xn, yn, zn = _D65
    fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)

    l_val = 116 * fy - 16
    a_val = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return np.stack([l_val, a_val, b_val], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    l_val, a_val, b_val = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l_val + 16) / 116
    fx = fy + a_val / 500
    fz = fy - b_val / 200

    xn, yn, zn = _D65
    x = _finv(fx) * xn
    y = _finv(fy) * yn
    z = _finv(fz) * zn

    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    return np.stack([_linear_to_srgb(r), _linear_to_srgb(g), _linear_to_srgb(b)], axis=-1)


def recolor(image: Image.Image, mask_img: Image.Image, target_lab: tuple[float, float, float]) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    lab = rgb_to_lab(rgb)

    mask_bool = np.asarray(mask_img.convert("L")) > 127
    if not np.any(mask_bool):
        return image.convert("RGB")

    l_channel = lab[..., 0]
    mean_l_in_mask = max(float(np.mean(l_channel[mask_bool])), 1e-3)

    target_l, target_a, target_b = target_lab
    l_scale = target_l / mean_l_in_mask

    l_new = np.clip(l_channel * l_scale, 0, 100)
    a_new = np.full_like(l_channel, target_a)
    b_new = np.full_like(l_channel, target_b)

    # preserve specular highlights: fade the chroma swap where the surface is already very bright
    highlight_threshold = np.percentile(l_channel[mask_bool], SPECULAR_PERCENTILE)
    specular = l_channel > highlight_threshold
    a_new = np.where(specular, lab[..., 1] * (1 - SPECULAR_CHROMA_FACTOR) + a_new * SPECULAR_CHROMA_FACTOR, a_new)
    b_new = np.where(specular, lab[..., 2] * (1 - SPECULAR_CHROMA_FACTOR) + b_new * SPECULAR_CHROMA_FACTOR, b_new)

    out_lab = np.stack([l_new, a_new, b_new], axis=-1)
    recolored_rgb = lab_to_rgb(out_lab)

    # feather the mask edge so the color change blends instead of a hard cutout
    feather = mask_img.convert("L").filter(ImageFilter.GaussianBlur(FEATHER_RADIUS))
    alpha = np.asarray(feather, dtype=np.float32)[..., np.newaxis] / 255.0

    blended = np.clip(rgb * (1 - alpha) + recolored_rgb * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")
