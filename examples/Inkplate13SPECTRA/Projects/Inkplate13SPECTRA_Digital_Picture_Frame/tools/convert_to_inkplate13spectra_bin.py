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
DEFAULT_FIT = "contain"
DEFAULT_DITHER = "stucki"
DEFAULT_BRIGHTNESS = 1.0
DEFAULT_CONTRAST = 1.10
DEFAULT_SATURATION = 1.16
DEFAULT_SHARPEN = 0.50
DEFAULT_DIFFUSION_STRENGTH = 0.60
DEFAULT_NEUTRAL_THRESHOLD = 18
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


def nominal_palette_rgb() -> list[tuple[int, int, int]]:
    return [NOMINAL_SPECTRA_PALETTE[name] for name in PANEL_PALETTE_ORDER]


def load_panel_palette(profile_path: str | None) -> tuple[list[tuple[int, int, int]], list[tuple[float, float, float]]]:
    """Load measured panel primaries from panel_profile.py's JSON output."""

    if not profile_path:
        palette_rgb = nominal_palette_rgb()
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


def adjust_image(
    image: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
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


def build_nearest_palette_lut(
    palette_lab: list[tuple[float, float, float]],
    allowed_indexes: tuple[int, ...] | None = None,
) -> bytearray:
    indexes = allowed_indexes or tuple(range(len(palette_lab)))
    lut = bytearray(32 * 32 * 32)

    for ri in range(32):
        r = 255 if ri == 31 else ri * 8 + 4
        for gi in range(32):
            g = 255 if gi == 31 else gi * 8 + 4
            for bi in range(32):
                b = 255 if bi == 31 else bi * 8 + 4
                lab = rgb_to_lab((r, g, b))
                best_index = indexes[0]
                best_distance = float("inf")
                for index in indexes:
                    pal_lab = palette_lab[index]
                    dl = lab[0] - pal_lab[0]
                    da = lab[1] - pal_lab[1]
                    db = lab[2] - pal_lab[2]
                    distance = dl * dl + da * da + db * db
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


def indexed_image_from_bytes(
    indices: bytes | bytearray,
    size: tuple[int, int],
    palette_rgb: list[tuple[int, int, int]],
) -> Image.Image:
    image = Image.frombytes("P", size, bytes(indices))
    image.putpalette(palette_values(palette_rgb))
    return image


def is_neutral_rgb(r: int, g: int, b: int, neutral_threshold: int) -> bool:
    return neutral_threshold > 0 and max(r, g, b) - min(r, g, b) <= neutral_threshold


def neutralize_error(err_r: int | float, err_g: int | float, err_b: int | float) -> tuple[float, float, float]:
    gray_error = (float(err_r) + float(err_g) + float(err_b)) / 3.0
    return gray_error, gray_error, gray_error


def nearest_palette_quantize(
    image: Image.Image,
    palette_rgb: list[tuple[int, int, int]],
    palette_lut: bytearray,
    neutral_lut: bytearray,
    neutral_threshold: int,
) -> Image.Image:
    data = image.tobytes()
    out = bytearray(image.width * image.height)
    for pixel in range(image.width * image.height):
        base = pixel * 3
        r, g, b = data[base], data[base + 1], data[base + 2]
        lut = neutral_lut if is_neutral_rgb(r, g, b, neutral_threshold) else palette_lut
        out[pixel] = palette_index_for_rgb(r, g, b, lut)
    return indexed_image_from_bytes(out, image.size, palette_rgb)


def error_diffusion_quantize(
    image: Image.Image,
    kernel_name: str,
    palette_rgb: list[tuple[int, int, int]],
    palette_lut: bytearray,
    neutral_lut: bytearray,
    neutral_threshold: int,
    diffusion_strength: float,
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

            neutral = is_neutral_rgb(r, g, b, neutral_threshold)
            lut = neutral_lut if neutral else palette_lut
            index = palette_index_for_rgb(r, g, b, lut)
            out[y * width + x] = index

            pal_r, pal_g, pal_b = palette_rgb[index]
            err_r = r - pal_r
            err_g = g - pal_g
            err_b = b - pal_b
            if neutral:
                err_r, err_g, err_b = neutralize_error(err_r, err_g, err_b)

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
    neutral_threshold: int,
    diffusion_strength: float,
    serpentine: bool,
) -> Image.Image:
    if dither == "none":
        return nearest_palette_quantize(image, palette_rgb, palette_lut, neutral_lut, neutral_threshold)
    if dither == "pillow":
        return image.quantize(palette=build_palette_image(palette_rgb), dither=Image.Dither.FLOYDSTEINBERG)
    return error_diffusion_quantize(
        image,
        dither,
        palette_rgb,
        palette_lut,
        neutral_lut,
        neutral_threshold,
        diffusion_strength,
        serpentine,
    )


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
        "--dither",
        choices=("stucki", "floyd-steinberg", "atkinson", "pillow", "none"),
        default=DEFAULT_DITHER,
        help="Palette dithering method. stucki is the default quality-oriented renderer.",
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
        "--neutral-threshold",
        type=int,
        default=DEFAULT_NEUTRAL_THRESHOLD,
        help="RGB max-min threshold for neutral pixels that should dither with black/white only. Use 0 to disable.",
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
        sharpen=args.sharpen,
        autocontrast=args.autocontrast,
        autocontrast_cutoff=args.autocontrast_cutoff,
    )
    quantized = quantize_to_spectra(
        adjusted,
        dither=args.dither,
        palette_rgb=palette_rgb,
        palette_lut=palette_lut,
        neutral_lut=neutral_lut,
        neutral_threshold=args.neutral_threshold,
        diffusion_strength=args.diffusion_strength,
        serpentine=not args.no_serpentine,
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

    palette_rgb, palette_lab = load_panel_palette(args.panel_profile)
    palette_lut = build_nearest_palette_lut(palette_lab)
    neutral_lut = build_nearest_palette_lut(palette_lab, allowed_indexes=(0, 1))

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
            convert_one(input_path, output_path, preview_path, args, palette_rgb, palette_lut, neutral_lut)
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
