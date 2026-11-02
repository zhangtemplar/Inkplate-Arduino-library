#!/usr/bin/env python3
"""Convert common image files to Inkplate 13SPECTRA packed BIN frames.

The output format starts with a 32 byte little-endian header followed by the
native Inkplate 13SPECTRA framebuffer: 1200 x 1600 pixels, two 4-bit Spectra 6
color indexes per byte, high nibble first. Source images can be packed as either
1600 x 1200 landscape or 1200 x 1600 portrait logical views.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import sys
from pathlib import Path

try:
    from PIL import Image, ImageColor, ImageEnhance, ImageFilter, ImageOps
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: pip3 install Pillow") from exc


MAGIC = b"IP13SP6B"
VERSION = 1
LANDSCAPE_SIZE = (1600, 1200)
PORTRAIT_SIZE = (1200, 1600)
NATIVE_WIDTH = 1200
NATIVE_HEIGHT = 1600
DATA_BYTES = NATIVE_WIDTH * NATIVE_HEIGHT // 2
INPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PANEL_PALETTE_ORDER = ("black", "white", "yellow", "red", "blue", "green")
DEFAULT_PALETTE_PRESET = "photo-frame"
DEFAULT_FIT = "contain"
DEFAULT_DITHER = "stucki"
DEFAULT_BRIGHTNESS = 1.0
DEFAULT_CONTRAST = 1.10
DEFAULT_SATURATION = 1.0
DEFAULT_RED_SATURATION_SCALE = 0.85
DEFAULT_SHARPEN = 0.50
DEFAULT_COLOR_MATCHING = "lab"
DEFAULT_TONE_CURVE = "off"
DEFAULT_TONE_CURVE_STRENGTH = 0.0
DEFAULT_SHADOW_BOOST = 0.0
DEFAULT_HIGHLIGHT_COMPRESS = 0.0
DEFAULT_TONE_MIDPOINT = 0.5
DEFAULT_DYNAMIC_RANGE = "display"
DEFAULT_DYNAMIC_RANGE_STRENGTH = 0.25
DEFAULT_RANGE_LOW_PERCENTILE = 0.01
DEFAULT_RANGE_HIGH_PERCENTILE = 0.99
DEFAULT_WHITE_PRESERVE_PERCENTILE = 0.99
DEFAULT_WHITE_PRESERVE_MIN_LUMA = 180
DEFAULT_WHITE_PRESERVE_MAX_SATURATION = 0.18
DEFAULT_DIFFUSION_STRENGTH = 0.35
DEFAULT_CHROMA_DIFFUSION_STRENGTH = 0.10
DEFAULT_NEUTRAL_THRESHOLD = 22
DEFAULT_NEUTRAL_SATURATION = 0.26
DEFAULT_SHADOW_NEUTRAL_MAX = 88
DEFAULT_SHADOW_NEUTRAL_SATURATION = 0.42
DEFAULT_EDGE_CLEANUP_STRENGTH = 0.25
DEFAULT_EDGE_CLEANUP_THRESHOLD = 42
ROTATION_BY_ORIENTATION = {
    "portrait": 0,
    "landscape": 1,
}

NOMINAL_SPECTRA_PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
}

CALIBRATED_SPECTRA_PALETTE_PRESETS = {
    "nominal": NOMINAL_SPECTRA_PALETTE,
    # Palettes adapted from paperlesspaper/epdoptimize. These are calibrated display
    # appearance colors used for matching; the packed BIN still stores native Inkplate indexes.
    # photo-frame keeps calibrated non-red inks but treats red as a strong ink so warm
    # skin, pink, and orange tones do not collapse into too much red.
    "photo-frame": {
        "black": (31, 34, 38),
        "white": (214, 214, 214),
        "yellow": (219, 213, 41),
        "red": (255, 32, 32),
        "blue": (65, 108, 225),
        "green": (6, 116, 6),
    },
    "epdoptimize-spectra6": {
        "black": (31, 34, 38),
        "white": (185, 199, 201),
        "yellow": (193, 187, 30),
        "red": (98, 32, 30),
        "blue": (35, 63, 142),
        "green": (53, 86, 58),
    },
    "epdoptimize-aitjcize": {
        "black": (2, 2, 2),
        "white": (190, 200, 200),
        "yellow": (205, 202, 0),
        "red": (135, 19, 0),
        "blue": (5, 64, 158),
        "green": (39, 102, 60),
    },
    "epdoptimize-boeber": {
        "black": (31, 34, 38),
        "white": (214, 214, 214),
        "yellow": (219, 213, 41),
        "red": (234, 72, 67),
        "blue": (65, 108, 225),
        "green": (6, 116, 6),
    },
    "epdoptimize-legacy": {
        "black": (25, 30, 33),
        "white": (232, 232, 232),
        "yellow": (239, 222, 68),
        "red": (178, 19, 24),
        "blue": (33, 87, 186),
        "green": (18, 95, 32),
    },
}

DIFFUSION_KERNELS = {
    "floyd-steinberg": (
        16,
        [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)],
    ),
    "atkinson": (
        8,
        [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)],
    ),
    "stucki": (
        42,
        [
            (1, 0, 8),
            (2, 0, 4),
            (-2, 1, 2),
            (-1, 1, 4),
            (0, 1, 8),
            (1, 1, 4),
            (2, 1, 2),
            (-2, 2, 1),
            (-1, 2, 2),
            (0, 2, 4),
            (1, 2, 2),
            (2, 2, 1),
        ],
    ),
}

def build_palette_image(palette_rgb: list[tuple[int, int, int]]) -> Image.Image:
    pal = Image.new("P", (1, 1))
    pal.putpalette(palette_values(palette_rgb))
    return pal


def palette_values(palette_rgb: list[tuple[int, int, int]]) -> list[int]:
    palette = []
    for rgb in palette_rgb:
        palette.extend(rgb)
    palette.extend([255, 255, 255] * (256 - len(palette_rgb)))
    return palette


def preset_palette_rgb(preset: str) -> list[tuple[int, int, int]]:
    if preset not in CALIBRATED_SPECTRA_PALETTE_PRESETS:
        choices = ", ".join(sorted(CALIBRATED_SPECTRA_PALETTE_PRESETS))
        raise ValueError(f"Unknown palette preset {preset!r}. Choices: {choices}")
    palette = CALIBRATED_SPECTRA_PALETTE_PRESETS[preset]
    return [palette[name] for name in PANEL_PALETTE_ORDER]


def load_panel_palette(
    profile_path: str | None,
    palette_preset: str,
) -> tuple[list[tuple[int, int, int]], list[tuple[float, float, float]]]:
    """Load measured panel primaries from panel_profile.py's JSON output."""

    if not profile_path:
        palette_rgb = preset_palette_rgb(palette_preset)
        return palette_rgb, [rgb_to_lab(rgb) for rgb in palette_rgb]

    path = expand_path(profile_path)
    if path.suffix.lower() == ".py":
        raise ValueError(
            "--panel-profile expects the measured JSON output from panel_profile.py, "
            "for example spectra6_profile.json, not panel_profile.py itself."
        )

    profile = json.loads(path.read_text())
    primaries = {str(primary["name"]).lower(): primary for primary in profile.get("primaries", [])}

    missing = [name for name in PANEL_PALETTE_ORDER if name not in primaries]
    if missing:
        raise ValueError(f"Panel profile is missing primaries: {', '.join(missing)}")

    palette_rgb: list[tuple[int, int, int]] = []
    palette_lab: list[tuple[float, float, float]] = []
    for name in PANEL_PALETTE_ORDER:
        primary = primaries[name]
        rgb = primary.get("srgb8") or primary.get("nominal_srgb8") or NOMINAL_SPECTRA_PALETTE[name]
        lab = primary.get("Lab_D65")

        if len(rgb) != 3:
            raise ValueError(f"Panel profile primary {name} has invalid srgb8")
        palette_rgb.append(tuple(max(0, min(255, int(round(channel)))) for channel in rgb))

        if lab is not None:
            if len(lab) != 3:
                raise ValueError(f"Panel profile primary {name} has invalid Lab_D65")
            palette_lab.append(tuple(float(channel) for channel in lab))
        else:
            palette_lab.append(rgb_to_lab(palette_rgb[-1]))

    return palette_rgb, palette_lab


def choose_orientation(image: Image.Image, requested: str) -> str:
    if requested != "auto":
        return requested
    return "portrait" if image.height > image.width else "landscape"


def target_size_for_orientation(orientation: str) -> tuple[int, int]:
    return PORTRAIT_SIZE if orientation == "portrait" else LANDSCAPE_SIZE


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def has_glob(value: str) -> bool:
    return any(ch in value for ch in "*?[")


def is_supported_input(path: Path) -> bool:
    return path.suffix.lower() in INPUT_EXTENSIONS


def split_legacy_output(inputs: list[str], output: str | None) -> tuple[list[str], str | None]:
    """Keep the old `script input.jpg output.bin` command working."""

    if output is not None or len(inputs) != 2:
        return inputs, output

    first = expand_path(inputs[0])
    second = expand_path(inputs[1])
    if not has_glob(inputs[0]) and not first.is_dir() and second.suffix.lower() == ".bin":
        return [inputs[0]], inputs[1]
    return inputs, output


def collect_input_files(inputs: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []

    for raw in inputs:
        expanded_raw = os.path.expandvars(os.path.expanduser(raw))
        if has_glob(expanded_raw):
            matches = [Path(match) for match in glob.glob(expanded_raw, recursive=recursive)]
            if not matches:
                raise FileNotFoundError(f"No files matched {raw}")
            files.extend(path for path in matches if path.is_file() and is_supported_input(path))
            continue

        path = Path(expanded_raw)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(child for child in path.glob(pattern) if child.is_file() and is_supported_input(child))
            continue

        if path.is_file() and is_supported_input(path):
            files.append(path)
            continue

        if path.exists():
            raise ValueError(f"Unsupported input type or extension: {path}")
        raise FileNotFoundError(f"Input not found: {path}")

    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            unique.append(path)
            seen.add(key)

    return sorted(unique, key=lambda p: str(p).lower())


def input_specs_are_batch(inputs: list[str], output: str | None, output_dir: str | None) -> bool:
    if output_dir is not None:
        return True
    if output is not None and expand_path(output).suffix.lower() != ".bin":
        return True
    if len(inputs) > 1:
        return True
    if not inputs:
        return False

    raw = inputs[0]
    return has_glob(raw) or expand_path(raw).is_dir()


def output_path_for(input_path: Path, output: str | None, output_dir: str | None, input_count: int) -> Path:
    if output_dir:
        return expand_path(output_dir) / input_path.with_suffix(".bin").name

    if output:
        resolved = expand_path(output)
        if input_count == 1 and resolved.suffix.lower() == ".bin":
            return resolved
        return resolved / input_path.with_suffix(".bin").name

    return input_path.with_suffix(".bin")


def preview_path_for(input_path: Path, preview: str | None, input_count: int) -> Path | None:
    if not preview:
        return None

    resolved = expand_path(preview)
    if input_count == 1 and resolved.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return resolved
    return resolved / f"{input_path.stem}_preview.png"


def open_rgb(path: Path, background: tuple[int, int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path))
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*background, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return image.convert("RGB")


def fit_image(
    image: Image.Image,
    target: tuple[int, int],
    fit: str,
    background: tuple[int, int, int],
) -> Image.Image:
    if fit == "cover":
        return ImageOps.fit(image, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    if fit == "stretch":
        return image.resize(target, Image.Resampling.LANCZOS)

    image = image.copy()
    image.thumbnail(target, Image.Resampling.LANCZOS, reducing_gap=3.0)
    canvas = Image.new("RGB", target, background)
    x = (target[0] - image.width) // 2
    y = (target[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def clamp_float(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def relative_luma_rgb(r: int | float, g: int | float, b: int | float) -> float:
    return 0.299 * float(r) + 0.587 * float(g) + 0.114 * float(b)


def histogram_percentile(histogram: list[int], percentile: float) -> int:
    percentile = clamp_float(percentile, 0.0, 1.0)
    total = sum(histogram)
    if total <= 0:
        return 0

    target = max(1, int(math.ceil(total * percentile)))
    seen = 0
    for value, count in enumerate(histogram):
        seen += count
        if seen >= target:
            return value
    return 255


def apply_scurve_tone(
    image: Image.Image,
    tone_curve: str,
    strength: float,
    shadow_boost: float,
    highlight_compress: float,
    midpoint: float,
) -> Image.Image:
    if tone_curve == "off" and shadow_boost == 0 and highlight_compress == 0:
        return image

    strength = clamp_float(strength, 0.0, 1.0)
    midpoint = clamp_float(midpoint, 0.05, 0.95)
    shadow_boost = clamp_float(shadow_boost, -1.0, 1.0)
    highlight_compress = clamp_float(highlight_compress, -2.0, 2.0)
    lut: list[int] = []

    for value in range(256):
        x = value / 255.0
        y = x

        if tone_curve == "scurve" and strength > 0:
            if x < midpoint:
                lower = x / midpoint
                curved = midpoint * lower * lower * (3.0 - 2.0 * lower)
            else:
                upper = (x - midpoint) / (1.0 - midpoint)
                curved = midpoint + (1.0 - midpoint) * upper * upper * (3.0 - 2.0 * upper)
            y += (curved - x) * strength

        if shadow_boost:
            y += shadow_boost * (1.0 - x) * (1.0 - x) * x
        if highlight_compress:
            y -= highlight_compress * x * x * (1.0 - x)

        lut.append(clamp_byte(y * 255.0))

    return image.point(lut * 3)


def apply_dynamic_range_compression(
    image: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    mode: str,
    strength: float,
    low_percentile: float,
    high_percentile: float,
    preserve_white: bool,
    white_preserve_percentile: float,
    white_preserve_min_luma: int,
    white_preserve_max_saturation: float,
) -> Image.Image:
    if mode == "off" or strength <= 0:
        return image

    strength = clamp_float(strength, 0.0, 1.0)
    palette_lumas = [relative_luma_rgb(*rgb) for rgb in palette_rgb]
    target_low = min(palette_lumas)
    target_high = max(palette_lumas)
    if target_high - target_low < 1:
        return image

    gray = image.convert("L")
    histogram = gray.histogram()
    if mode == "auto":
        source_low = histogram_percentile(histogram, low_percentile)
        source_high = histogram_percentile(histogram, high_percentile)
    else:
        source_low = 0
        source_high = 255
    if source_high <= source_low:
        source_low, source_high = 0, 255

    white_luma = relative_luma_rgb(*palette_rgb[1])
    white_threshold = max(
        white_preserve_min_luma,
        histogram_percentile(histogram, white_preserve_percentile),
    )

    data = bytearray(image.tobytes())
    for base in range(0, len(data), 3):
        r = data[base]
        g = data[base + 1]
        b = data[base + 2]
        luma = relative_luma_rgb(r, g, b)
        normalized = clamp_float((luma - source_low) / (source_high - source_low), 0.0, 1.0)
        fitted_luma = target_low + normalized * (target_high - target_low)
        new_luma = luma + (fitted_luma - luma) * strength

        if (
            preserve_white
            and luma >= white_threshold
            and rgb_saturation(r, g, b) <= white_preserve_max_saturation
            and new_luma < white_luma
        ):
            new_luma = white_luma

        delta = new_luma - luma
        data[base] = clamp_byte(r + delta)
        data[base + 1] = clamp_byte(g + delta)
        data[base + 2] = clamp_byte(b + delta)

    return Image.frombytes("RGB", image.size, bytes(data))


def reduce_red_saturation(image: Image.Image, scale: float) -> Image.Image:
    """Slightly compress dominant red chroma to avoid red ink over-selection."""

    if scale >= 0.999:
        return image
    scale = max(0.0, min(1.0, scale))

    data = bytearray(image.tobytes())
    for base in range(0, len(data), 3):
        r = data[base]
        g = data[base + 1]
        b = data[base + 2]
        other = max(g, b)
        red_excess = r - other
        if r < 72 or red_excess <= 16 or r < g + 10 or r < b + 14:
            continue

        strength = min(1.0, (red_excess - 16) / 96.0)
        effective_scale = 1.0 - (1.0 - scale) * strength
        data[base] = clamp_byte(other + 16 + (red_excess - 16) * effective_scale)

    return Image.frombytes("RGB", image.size, bytes(data))


def adjust_image(
    image: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
    red_saturation_scale: float,
    tone_curve: str,
    tone_curve_strength: float,
    shadow_boost: float,
    highlight_compress: float,
    tone_midpoint: float,
    sharpen: float,
    autocontrast: bool,
    autocontrast_cutoff: float,
) -> Image.Image:
    if autocontrast:
        image = ImageOps.autocontrast(image, cutoff=autocontrast_cutoff)
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    if saturation != 1.0:
        image = ImageEnhance.Color(image).enhance(saturation)
    image = apply_scurve_tone(image, tone_curve, tone_curve_strength, shadow_boost, highlight_compress, tone_midpoint)
    image = reduce_red_saturation(image, red_saturation_scale)
    if sharpen > 0:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen * 100), threshold=2))
    return image


def srgb_to_linear(value: float) -> float:
    value /= 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def xyz_to_lab_component(value: float) -> float:
    if value > 0.008856:
        return value ** (1.0 / 3.0)
    return 7.787 * value + 16.0 / 116.0


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r = srgb_to_linear(rgb[0])
    g = srgb_to_linear(rgb[1])
    b = srgb_to_linear(rgb[2])

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    fx = xyz_to_lab_component(x / 0.95047)
    fy = xyz_to_lab_component(y)
    fz = xyz_to_lab_component(z / 1.08883)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def rgb_saturation(r: int, g: int, b: int) -> float:
    value = max(r, g, b)
    if value <= 0:
        return 0.0
    return (value - min(r, g, b)) / value


def rgb_hue(r: int, g: int, b: int) -> float:
    red = r / 255.0
    green = g / 255.0
    blue = b / 255.0
    high = max(red, green, blue)
    low = min(red, green, blue)
    delta = high - low
    if delta <= 0:
        return 0.0

    if high == red:
        hue = 60.0 * (((green - blue) / delta) % 6.0)
    elif high == green:
        hue = 60.0 * ((blue - red) / delta + 2.0)
    else:
        hue = 60.0 * ((red - green) / delta + 4.0)
    return hue + 360.0 if hue < 0 else hue


def hue_distance(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def palette_distance(
    rgb: tuple[int, int, int],
    index: int,
    palette_rgb: list[tuple[int, int, int]],
    palette_lab: list[tuple[float, float, float]],
    palette_saturations: list[float],
    palette_hues: list[float],
    color_matching: str,
) -> float:
    r, g, b = rgb
    pal_r, pal_g, pal_b = palette_rgb[index]

    if color_matching == "lab":
        lab = rgb_to_lab(rgb)
        pal_lab = palette_lab[index]
        dl = lab[0] - pal_lab[0]
        da = lab[1] - pal_lab[1]
        db = lab[2] - pal_lab[2]
        return dl * dl + da * da + db * db + scarce_ink_penalty(r, g, b, index, index, 1.0) * 0.12

    dr = pal_r - r
    dg = pal_g - g
    db = pal_b - b
    distance = dr * dr + dg * dg + db * db
    if color_matching != "chroma":
        return distance + scarce_ink_penalty(r, g, b, index, index, 1.0) * 0.35

    source_saturation = rgb_saturation(r, g, b)
    if source_saturation < 0.12:
        return distance

    penalty = 0.0
    palette_saturation = palette_saturations[index]
    if palette_saturation <= 0.12:
        penalty += min(330.0, source_saturation * 1300.0)
    else:
        penalty += hue_distance(rgb_hue(r, g, b), palette_hues[index]) * 3.0

    return math.sqrt(distance) + penalty + scarce_ink_penalty(r, g, b, index, index, 1.0) * 0.004


def build_nearest_palette_lut(
    palette_rgb: list[tuple[int, int, int]],
    palette_lab: list[tuple[float, float, float]],
    color_matching: str,
    allowed_indexes: tuple[int, ...] | None = None,
) -> bytearray:
    indexes = allowed_indexes or tuple(range(len(palette_lab)))
    lut = bytearray(32 * 32 * 32)
    palette_saturations = [rgb_saturation(*rgb) for rgb in palette_rgb]
    palette_hues = [rgb_hue(*rgb) for rgb in palette_rgb]

    for ri in range(32):
        r = 255 if ri == 31 else ri * 8 + 4
        for gi in range(32):
            g = 255 if gi == 31 else gi * 8 + 4
            for bi in range(32):
                b = 255 if bi == 31 else bi * 8 + 4
                best_index = indexes[0]
                best_distance = float("inf")
                rgb = (r, g, b)
                for index in indexes:
                    distance = palette_distance(
                        rgb,
                        index,
                        palette_rgb,
                        palette_lab,
                        palette_saturations,
                        palette_hues,
                        color_matching,
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_index = index
                lut[(ri << 10) | (gi << 5) | bi] = best_index

    return lut


def clamp_byte(value: float) -> int:
    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return int(value + 0.5)


def palette_index_for_rgb(r: int, g: int, b: int, palette_lut: bytearray) -> int:
    return palette_lut[((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)]


def scarce_ink_penalty(
    r: int,
    g: int,
    b: int,
    idx_a: int,
    idx_b: int,
    t: float,
) -> float:
    red_coverage = (1.0 - t if idx_a == 3 else 0.0) + (t if idx_b == 3 else 0.0)
    blue_coverage = (1.0 - t if idx_a == 4 else 0.0) + (t if idx_b == 4 else 0.0)
    black_coverage = (1.0 - t if idx_a == 0 else 0.0) + (t if idx_b == 0 else 0.0)

    red_allow = clamp_float((r - max(g, b) - 24) / 96.0, 0.0, 1.0)
    blue_allow = clamp_float((b - max(r, g) - 24) / 96.0, 0.0, 1.0)
    luma = relative_luma_rgb(r, g, b)
    black_penalty = black_coverage * clamp_float((luma - 48.0) / 128.0, 0.0, 1.0) * 7000.0

    return (
        red_coverage * (1.0 - red_allow) * 18000.0
        + blue_coverage * (1.0 - blue_allow) * 6000.0
        + black_penalty
    )


def build_pair_mix_lut(
    palette_rgb: list[tuple[int, int, int]],
    allowed_indexes: tuple[int, ...] | None = None,
) -> tuple[bytearray, bytearray, bytearray]:
    indexes = allowed_indexes or tuple(range(len(palette_rgb)))
    first = bytearray(32 * 32 * 32)
    second = bytearray(32 * 32 * 32)
    coverage = bytearray(32 * 32 * 32)
    pairs = [(a, b) for pair_index, a in enumerate(indexes) for b in indexes[pair_index:]]

    for ri in range(32):
        r = 255 if ri == 31 else ri * 8 + 4
        for gi in range(32):
            g = 255 if gi == 31 else gi * 8 + 4
            for bi in range(32):
                b = 255 if bi == 31 else bi * 8 + 4
                best_first = indexes[0]
                best_second = indexes[0]
                best_coverage = 0
                best_distance = float("inf")

                for idx_a, idx_b in pairs:
                    ar, ag, ab = palette_rgb[idx_a]
                    br, bg, bb = palette_rgb[idx_b]
                    vr = br - ar
                    vg = bg - ag
                    vb = bb - ab
                    denom = vr * vr + vg * vg + vb * vb
                    if denom <= 0:
                        t = 0.0
                    else:
                        t = ((r - ar) * vr + (g - ag) * vg + (b - ab) * vb) / denom
                        t = clamp_float(t, 0.0, 1.0)

                    mix_r = ar + vr * t
                    mix_g = ag + vg * t
                    mix_b = ab + vb * t
                    dr = r - mix_r
                    dg = g - mix_g
                    db = b - mix_b
                    distance = dr * dr + dg * dg + db * db + scarce_ink_penalty(r, g, b, idx_a, idx_b, t)

                    if distance < best_distance:
                        best_distance = distance
                        best_first = idx_a
                        best_second = idx_b
                        best_coverage = clamp_byte(t * 255.0)

                lut_index = (ri << 10) | (gi << 5) | bi
                first[lut_index] = best_first
                second[lut_index] = best_second
                coverage[lut_index] = best_coverage

    return first, second, coverage


def pair_mix_for_rgb(
    r: int,
    g: int,
    b: int,
    mix_lut: tuple[bytearray, bytearray, bytearray],
) -> tuple[int, int, int]:
    lut_index = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
    return mix_lut[0][lut_index], mix_lut[1][lut_index], mix_lut[2][lut_index]


def coverage_threshold(x: int, y: int) -> int:
    value = (x * 374761393 + y * 668265263 + 0x9E3779B9) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 1274126177) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFF


def indexed_image_from_bytes(
    indices: bytes | bytearray,
    size: tuple[int, int],
    palette_rgb: list[tuple[int, int, int]],
) -> Image.Image:
    image = Image.frombytes("P", size, bytes(indices))
    image.putpalette(palette_values(palette_rgb))
    return image


def is_neutral_rgb(
    r: int,
    g: int,
    b: int,
    neutral_threshold: int,
    neutral_saturation: float,
    shadow_neutral_max: int,
    shadow_neutral_saturation: float,
) -> bool:
    chroma = max(r, g, b) - min(r, g, b)
    if neutral_threshold > 0 and chroma <= neutral_threshold:
        return True

    saturation = rgb_saturation(r, g, b)
    if neutral_saturation > 0 and saturation <= neutral_saturation:
        return True

    value = max(r, g, b)
    return (
        shadow_neutral_max > 0
        and value <= shadow_neutral_max
        and shadow_neutral_saturation > 0
        and saturation <= shadow_neutral_saturation
    )


def neutralize_error(err_r: int | float, err_g: int | float, err_b: int | float) -> tuple[float, float, float]:
    gray_error = 0.299 * float(err_r) + 0.587 * float(err_g) + 0.114 * float(err_b)
    return gray_error, gray_error, gray_error


def damp_chroma_error(
    err_r: int | float,
    err_g: int | float,
    err_b: int | float,
    chroma_strength: float,
) -> tuple[float, float, float]:
    if chroma_strength >= 0.999:
        return float(err_r), float(err_g), float(err_b)
    if chroma_strength <= 0:
        return neutralize_error(err_r, err_g, err_b)

    gray_error = 0.299 * float(err_r) + 0.587 * float(err_g) + 0.114 * float(err_b)
    return (
        gray_error + (float(err_r) - gray_error) * chroma_strength,
        gray_error + (float(err_g) - gray_error) * chroma_strength,
        gray_error + (float(err_b) - gray_error) * chroma_strength,
    )


def nearest_palette_quantize(
    image: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    palette_lut: bytearray,
    neutral_lut: bytearray,
    neutral_threshold: int,
    neutral_saturation: float,
    shadow_neutral_max: int,
    shadow_neutral_saturation: float,
) -> Image.Image:
    data = image.tobytes()
    out = bytearray(image.width * image.height)
    for pixel in range(image.width * image.height):
        base = pixel * 3
        r, g, b = data[base], data[base + 1], data[base + 2]
        lut = (
            neutral_lut
            if is_neutral_rgb(
                r,
                g,
                b,
                neutral_threshold,
                neutral_saturation,
                shadow_neutral_max,
                shadow_neutral_saturation,
            )
            else palette_lut
        )
        out[pixel] = palette_index_for_rgb(r, g, b, lut)
    return indexed_image_from_bytes(out, image.size, palette_rgb)


def coverage_quantize(
    image: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    mix_lut: tuple[bytearray, bytearray, bytearray],
    neutral_lut: bytearray,
    neutral_threshold: int,
    neutral_saturation: float,
    shadow_neutral_max: int,
    shadow_neutral_saturation: float,
) -> Image.Image:
    width, height = image.size
    data = image.tobytes()
    out = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            pixel = y * width + x
            base = pixel * 3
            r, g, b = data[base], data[base + 1], data[base + 2]
            if is_neutral_rgb(
                r,
                g,
                b,
                neutral_threshold,
                neutral_saturation,
                shadow_neutral_max,
                shadow_neutral_saturation,
            ):
                out[pixel] = palette_index_for_rgb(r, g, b, neutral_lut)
                continue

            first, second, coverage = pair_mix_for_rgb(r, g, b, mix_lut)
            out[pixel] = second if coverage_threshold(x, y) < coverage else first

    return indexed_image_from_bytes(out, image.size, palette_rgb)


def error_diffusion_quantize(
    image: Image.Image,
    kernel_name: str,
    palette_rgb: list[tuple[int, int, int]],
    palette_lut: bytearray,
    neutral_lut: bytearray,
    neutral_threshold: int,
    neutral_saturation: float,
    shadow_neutral_max: int,
    shadow_neutral_saturation: float,
    diffusion_strength: float,
    chroma_diffusion_strength: float,
    serpentine: bool,
) -> Image.Image:
    denominator, kernel = DIFFUSION_KERNELS[kernel_name]
    max_dy = max(dy for _, dy, _ in kernel)
    width, height = image.size
    data = image.tobytes()
    out = bytearray(width * height)
    error_rows = [[0.0] * (width * 3) for _ in range(max_dy + 1)]

    for y in range(height):
        current_errors = error_rows[0]
        right_to_left = serpentine and (y & 1)
        x_range = range(width - 1, -1, -1) if right_to_left else range(width)
        x_direction = -1 if right_to_left else 1

        for x in x_range:
            base = (y * width + x) * 3
            error_offset = x * 3
            r = clamp_byte(data[base] + current_errors[error_offset])
            g = clamp_byte(data[base + 1] + current_errors[error_offset + 1])
            b = clamp_byte(data[base + 2] + current_errors[error_offset + 2])

            neutral = is_neutral_rgb(
                r,
                g,
                b,
                neutral_threshold,
                neutral_saturation,
                shadow_neutral_max,
                shadow_neutral_saturation,
            )
            lut = neutral_lut if neutral else palette_lut
            index = palette_index_for_rgb(r, g, b, lut)
            out[y * width + x] = index

            pal_r, pal_g, pal_b = palette_rgb[index]
            err_r = r - pal_r
            err_g = g - pal_g
            err_b = b - pal_b
            if neutral:
                err_r, err_g, err_b = neutralize_error(err_r, err_g, err_b)
            else:
                err_r, err_g, err_b = damp_chroma_error(err_r, err_g, err_b, chroma_diffusion_strength)

            for dx, dy, weight in kernel:
                nx = x + dx * x_direction
                if nx < 0 or nx >= width or y + dy >= height:
                    continue

                target = error_rows[dy]
                target_offset = nx * 3
                factor = diffusion_strength * weight / denominator
                target[target_offset] += err_r * factor
                target[target_offset + 1] += err_g * factor
                target[target_offset + 2] += err_b * factor

        error_rows.pop(0)
        error_rows.append([0.0] * (width * 3))

    return indexed_image_from_bytes(out, image.size, palette_rgb)


def quantize_to_spectra(
    image: Image.Image,
    dither: str,
    palette_rgb: list[tuple[int, int, int]],
    palette_lut: bytearray,
    neutral_lut: bytearray,
    mix_lut: tuple[bytearray, bytearray, bytearray],
    neutral_threshold: int,
    neutral_saturation: float,
    shadow_neutral_max: int,
    shadow_neutral_saturation: float,
    diffusion_strength: float,
    chroma_diffusion_strength: float,
    serpentine: bool,
) -> Image.Image:
    if dither == "none":
        return nearest_palette_quantize(
            image,
            palette_rgb,
            palette_lut,
            neutral_lut,
            neutral_threshold,
            neutral_saturation,
            shadow_neutral_max,
            shadow_neutral_saturation,
        )
    if dither == "coverage":
        return coverage_quantize(
            image,
            palette_rgb,
            mix_lut,
            neutral_lut,
            neutral_threshold,
            neutral_saturation,
            shadow_neutral_max,
            shadow_neutral_saturation,
        )
    if dither == "pillow":
        return image.quantize(palette=build_palette_image(palette_rgb), dither=Image.Dither.FLOYDSTEINBERG)
    return error_diffusion_quantize(
        image,
        dither,
        palette_rgb,
        palette_lut,
        neutral_lut,
        neutral_threshold,
        neutral_saturation,
        shadow_neutral_max,
        shadow_neutral_saturation,
        diffusion_strength,
        chroma_diffusion_strength,
        serpentine,
    )


def cleanup_dithered_edges(
    source: Image.Image,
    quantized: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    palette_lut: bytearray,
    neutral_lut: bytearray,
    neutral_threshold: int,
    neutral_saturation: float,
    shadow_neutral_max: int,
    shadow_neutral_saturation: float,
    strength: float,
    threshold: int,
) -> Image.Image:
    strength = clamp_float(strength, 0.0, 1.0)
    if strength <= 0:
        return quantized

    width, height = source.size
    if width < 3 or height < 3:
        return quantized

    effective_threshold = threshold + int((1.0 - strength) * 96)
    gray = source.convert("L").tobytes()
    source_data = source.tobytes()
    out = bytearray(quantized.tobytes())

    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            pixel = row + x
            gradient = abs(gray[pixel + 1] - gray[pixel - 1]) + abs(gray[pixel + width] - gray[pixel - width])
            if gradient < effective_threshold:
                continue

            base = pixel * 3
            r = source_data[base]
            g = source_data[base + 1]
            b = source_data[base + 2]
            lut = (
                neutral_lut
                if is_neutral_rgb(
                    r,
                    g,
                    b,
                    neutral_threshold,
                    neutral_saturation,
                    shadow_neutral_max,
                    shadow_neutral_saturation,
                )
                else palette_lut
            )
            out[pixel] = palette_index_for_rgb(r, g, b, lut)

    return indexed_image_from_bytes(out, quantized.size, palette_rgb)


def pack_native_frame(logical: Image.Image, orientation: str) -> bytes:
    """Map the logical view to the framebuffer used by the matching sketch rotation."""

    pixels = logical.tobytes()
    logical_width, logical_height = logical.size
    out = bytearray(DATA_BYTES)
    rotation = ROTATION_BY_ORIENTATION[orientation]

    for y in range(logical_height):
        for x in range(logical_width):
            color = pixels[y * logical_width + x] & 0x0F

            if rotation == 1:
                # Landscape uses sketch rotation 1:
                # native_col = y, native_row = 1599 - x.
                native_col = y
                native_row = logical_width - x - 1
            else:
                # Portrait uses sketch rotation 0:
                # native_col = 1199 - x, native_row = 1599 - y.
                native_col = logical_width - x - 1
                native_row = logical_height - y - 1

            byte_index = native_row * (NATIVE_WIDTH // 2) + native_col // 2

            if native_col & 1:
                out[byte_index] = (out[byte_index] & 0xF0) | color
            else:
                out[byte_index] = (out[byte_index] & 0x0F) | (color << 4)

    return bytes(out)


def write_bin(output: Path, payload: bytes, logical_size: tuple[int, int], orientation: str) -> None:
    rotation = ROTATION_BY_ORIENTATION[orientation]
    flags = 0x01 | (rotation << 1)  # native packed data plus the sketch rotation used when packing
    header = struct.pack(
        "<8sBBBBHHHHI8s",
        MAGIC,
        VERSION,
        flags,
        len(PANEL_PALETTE_ORDER),
        0,
        logical_size[0],
        logical_size[1],
        NATIVE_WIDTH,
        NATIVE_HEIGHT,
        len(payload),
        b"\0" * 8,
    )
    output.write_bytes(header + payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input image files, directories, or glob patterns. Paths may be absolute, relative, or start with ~.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output .bin path for one input, or output directory for batch input.",
    )
    parser.add_argument("--output-dir", help="Output directory for batch conversion.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directories.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist. Batch conversion skips existing outputs by default.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs in batch conversion.")
    parser.add_argument(
        "--fit",
        choices=("cover", "contain", "stretch"),
        default=DEFAULT_FIT,
        help="Resize mode. contain preserves the full image; cover crops to fill the frame.",
    )
    parser.add_argument(
        "--background",
        default="#FFFFFF",
        help="Background color for --fit contain, for example #FFFFFF.",
    )
    parser.add_argument(
        "--orientation",
        choices=("auto", "landscape", "portrait"),
        default="auto",
        help="Output orientation. auto chooses portrait for taller source images.",
    )
    parser.add_argument(
        "--panel-profile",
        help="Measured JSON profile from panel_profile.py. Uses real Spectra 6 primary colors for quantization.",
    )
    parser.add_argument(
        "--palette-preset",
        choices=tuple(sorted(CALIBRATED_SPECTRA_PALETTE_PRESETS)),
        default=DEFAULT_PALETTE_PRESET,
        help="Built-in calibrated palette used when --panel-profile is not supplied.",
    )
    parser.add_argument(
        "--color-matching",
        choices=("rgb", "lab", "chroma"),
        default=DEFAULT_COLOR_MATCHING,
        help="Palette distance model. lab is the conservative photo default; rgb is punchier; chroma protects saturated pastels.",
    )
    parser.add_argument(
        "--dither",
        choices=("coverage", "stucki", "floyd-steinberg", "atkinson", "pillow", "none"),
        default=DEFAULT_DITHER,
        help="Palette dithering method. stucki is the default; coverage uses two-ink mixing for flat-color collapse.",
    )
    parser.add_argument("--no-dither", action="store_true", help="Alias for --dither none.")
    parser.add_argument(
        "--brightness",
        type=float,
        default=DEFAULT_BRIGHTNESS,
        help="Brightness multiplier before quantization.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=DEFAULT_CONTRAST,
        help="Contrast multiplier before quantization.",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=DEFAULT_SATURATION,
        help="Saturation multiplier before quantization.",
    )
    parser.add_argument(
        "--red-saturation-scale",
        type=float,
        default=DEFAULT_RED_SATURATION_SCALE,
        help="Scale strongly red-dominant chroma before quantization. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--tone-curve",
        choices=("off", "scurve"),
        default=DEFAULT_TONE_CURVE,
        help="Optional S-curve tone mapping before range compression.",
    )
    parser.add_argument(
        "--tone-curve-strength",
        type=float,
        default=DEFAULT_TONE_CURVE_STRENGTH,
        help="S-curve blend strength from 0 to 1.",
    )
    parser.add_argument(
        "--shadow-boost",
        type=float,
        default=DEFAULT_SHADOW_BOOST,
        help="Lift or lower shadows before quantization. Positive values lift shadows.",
    )
    parser.add_argument(
        "--highlight-compress",
        type=float,
        default=DEFAULT_HIGHLIGHT_COMPRESS,
        help="Compress or expand highlights before quantization. Positive values preserve highlight detail.",
    )
    parser.add_argument(
        "--tone-midpoint",
        type=float,
        default=DEFAULT_TONE_MIDPOINT,
        help="Midpoint used by --tone-curve scurve.",
    )
    parser.add_argument(
        "--dynamic-range",
        choices=("off", "display", "auto"),
        default=DEFAULT_DYNAMIC_RANGE,
        help="Compress source lightness into the usable display range. auto uses percentile clipping.",
    )
    parser.add_argument(
        "--dynamic-range-strength",
        type=float,
        default=DEFAULT_DYNAMIC_RANGE_STRENGTH,
        help="Blend strength for dynamic range compression from 0 to 1.",
    )
    parser.add_argument(
        "--range-low-percentile",
        type=float,
        default=DEFAULT_RANGE_LOW_PERCENTILE,
        help="Low percentile used by --dynamic-range auto.",
    )
    parser.add_argument(
        "--range-high-percentile",
        type=float,
        default=DEFAULT_RANGE_HIGH_PERCENTILE,
        help="High percentile used by --dynamic-range auto.",
    )
    parser.add_argument(
        "--no-preserve-white",
        action="store_true",
        help="Disable white background preservation during dynamic range compression.",
    )
    parser.add_argument(
        "--white-preserve-percentile",
        type=float,
        default=DEFAULT_WHITE_PRESERVE_PERCENTILE,
        help="Bright percentile treated as background white when preserving whites.",
    )
    parser.add_argument(
        "--white-preserve-min-luma",
        type=int,
        default=DEFAULT_WHITE_PRESERVE_MIN_LUMA,
        help="Minimum source luma eligible for white preservation.",
    )
    parser.add_argument(
        "--white-preserve-max-saturation",
        type=float,
        default=DEFAULT_WHITE_PRESERVE_MAX_SATURATION,
        help="Maximum source saturation eligible for white preservation.",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=DEFAULT_SHARPEN,
        help="Unsharp-mask strength after resize. Use 0 to disable.",
    )
    parser.add_argument(
        "--diffusion-strength",
        type=float,
        default=DEFAULT_DIFFUSION_STRENGTH,
        help="Scale error diffusion. Lower values look cleaner; 1.0 is mathematically stronger but noisier.",
    )
    parser.add_argument(
        "--chroma-diffusion-strength",
        type=float,
        default=DEFAULT_CHROMA_DIFFUSION_STRENGTH,
        help="Scale color-only diffusion error. Lower values reduce colored speckles while preserving tone.",
    )
    parser.add_argument(
        "--neutral-threshold",
        type=int,
        default=DEFAULT_NEUTRAL_THRESHOLD,
        help="RGB max-min threshold for neutral pixels that should dither with black/white only. Use 0 to disable.",
    )
    parser.add_argument(
        "--neutral-saturation",
        type=float,
        default=DEFAULT_NEUTRAL_SATURATION,
        help="HSV-style saturation threshold for low-color pixels that should dither with black/white only.",
    )
    parser.add_argument(
        "--shadow-neutral-max",
        type=int,
        default=DEFAULT_SHADOW_NEUTRAL_MAX,
        help="Treat shadows at or below this RGB value as neutral when saturation is low. Use 0 to disable.",
    )
    parser.add_argument(
        "--shadow-neutral-saturation",
        type=float,
        default=DEFAULT_SHADOW_NEUTRAL_SATURATION,
        help="Saturation limit used with --shadow-neutral-max to suppress colored shadow speckles.",
    )
    parser.add_argument(
        "--edge-cleanup-strength",
        type=float,
        default=DEFAULT_EDGE_CLEANUP_STRENGTH,
        help="Snap strong edges back to direct palette matches after dithering. Use 0 to disable.",
    )
    parser.add_argument(
        "--edge-cleanup-threshold",
        type=int,
        default=DEFAULT_EDGE_CLEANUP_THRESHOLD,
        help="Base luminance-gradient threshold for --edge-cleanup-strength.",
    )
    parser.add_argument(
        "--no-serpentine",
        action="store_true",
        help="Disable alternating left/right diffusion rows.",
    )
    parser.add_argument("--autocontrast", action="store_true", help="Apply light autocontrast before quantization.")
    parser.add_argument(
        "--autocontrast-cutoff",
        type=float,
        default=0.5,
        help="Percent clipped at each end when --autocontrast is used.",
    )
    parser.add_argument("--preview", help="Optional preview PNG path for one input, or preview directory for batch.")
    return parser.parse_args()


def convert_one(
    input_path: Path,
    output_path: Path,
    preview_path: Path | None,
    args: argparse.Namespace,
    palette_rgb: list[tuple[int, int, int]],
    palette_lut: bytearray,
    neutral_lut: bytearray,
    mix_lut: tuple[bytearray, bytearray, bytearray],
) -> None:
    background = ImageColor.getrgb(args.background)
    image = open_rgb(input_path, background)
    orientation = choose_orientation(image, args.orientation)
    target = target_size_for_orientation(orientation)
    fitted = fit_image(image, target, args.fit, background)
    adjusted = adjust_image(
        fitted,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        red_saturation_scale=args.red_saturation_scale,
        tone_curve=args.tone_curve,
        tone_curve_strength=args.tone_curve_strength,
        shadow_boost=args.shadow_boost,
        highlight_compress=args.highlight_compress,
        tone_midpoint=args.tone_midpoint,
        sharpen=args.sharpen,
        autocontrast=args.autocontrast,
        autocontrast_cutoff=args.autocontrast_cutoff,
    )
    processed = apply_dynamic_range_compression(
        adjusted,
        palette_rgb=palette_rgb,
        mode=args.dynamic_range,
        strength=args.dynamic_range_strength,
        low_percentile=args.range_low_percentile,
        high_percentile=args.range_high_percentile,
        preserve_white=not args.no_preserve_white,
        white_preserve_percentile=args.white_preserve_percentile,
        white_preserve_min_luma=args.white_preserve_min_luma,
        white_preserve_max_saturation=args.white_preserve_max_saturation,
    )
    quantized = quantize_to_spectra(
        processed,
        dither=args.dither,
        palette_rgb=palette_rgb,
        palette_lut=palette_lut,
        neutral_lut=neutral_lut,
        mix_lut=mix_lut,
        neutral_threshold=args.neutral_threshold,
        neutral_saturation=args.neutral_saturation,
        shadow_neutral_max=args.shadow_neutral_max,
        shadow_neutral_saturation=args.shadow_neutral_saturation,
        diffusion_strength=args.diffusion_strength,
        chroma_diffusion_strength=args.chroma_diffusion_strength,
        serpentine=not args.no_serpentine,
    )
    quantized = cleanup_dithered_edges(
        processed,
        quantized,
        palette_rgb=palette_rgb,
        palette_lut=palette_lut,
        neutral_lut=neutral_lut,
        neutral_threshold=args.neutral_threshold,
        neutral_saturation=args.neutral_saturation,
        shadow_neutral_max=args.shadow_neutral_max,
        shadow_neutral_saturation=args.shadow_neutral_saturation,
        strength=args.edge_cleanup_strength,
        threshold=args.edge_cleanup_threshold,
    )

    if preview_path:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        quantized.convert("RGB").save(preview_path)

    payload = pack_native_frame(quantized, orientation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bin(output_path, payload, target, orientation)
    print(f"Wrote {output_path} ({orientation}, {target[0]}x{target[1]}, {len(payload) + 32} bytes)")


def main() -> int:
    args = parse_args()
    args.inputs, args.output = split_legacy_output(args.inputs, args.output)
    if args.no_dither:
        args.dither = "none"

    input_files = collect_input_files(args.inputs, recursive=args.recursive)
    if not input_files:
        raise SystemExit("No supported input images found.")

    palette_rgb, palette_lab = load_panel_palette(args.panel_profile, args.palette_preset)
    palette_lut = build_nearest_palette_lut(palette_rgb, palette_lab, args.color_matching)
    neutral_lut = build_nearest_palette_lut(palette_rgb, palette_lab, args.color_matching, allowed_indexes=(0, 1))
    mix_lut = build_pair_mix_lut(palette_rgb)

    batch_mode = input_specs_are_batch(args.inputs, args.output, args.output_dir)
    should_skip_existing = False if args.force else args.skip_existing or batch_mode
    converted = 0
    skipped = 0
    failed = 0

    for input_path in input_files:
        output_path = output_path_for(input_path, args.output, args.output_dir, len(input_files))
        preview_path = preview_path_for(input_path, args.preview, len(input_files))

        if should_skip_existing and output_path.exists():
            print(f"Skipped existing {output_path}")
            skipped += 1
            continue

        try:
            convert_one(
                input_path,
                output_path,
                preview_path,
                args,
                palette_rgb,
                palette_lut,
                neutral_lut,
                mix_lut,
            )
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"Failed {input_path}: {exc}", file=sys.stderr)

    if len(input_files) > 1:
        print(f"Summary: converted={converted}, skipped={skipped}, failed={failed}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
