#!/usr/bin/env python3
"""Convert common image files to Inkplate 13SPECTRA packed BIN frames.

The output format starts with a 32 byte little-endian header followed by the
native Inkplate 13SPECTRA framebuffer: 1200 x 1600 pixels, two 4-bit Spectra 6
color indexes per byte, high nibble first.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

try:
    from PIL import Image, ImageColor, ImageOps
except ImportError as exc:
    raise SystemExit("Pillow is required. Install it with: pip3 install Pillow") from exc


MAGIC = b"IP13SP6B"
VERSION = 1
LOGICAL_WIDTH = 1600
LOGICAL_HEIGHT = 1200
NATIVE_WIDTH = 1200
NATIVE_HEIGHT = 1600
DATA_BYTES = NATIVE_WIDTH * NATIVE_HEIGHT // 2

SPECTRA_PALETTE = [
    (0, 0, 0),        # 0 black
    (255, 255, 255),  # 1 white
    (255, 255, 0),    # 2 yellow
    (255, 0, 0),      # 3 red
    (0, 0, 255),      # 4 blue
    (0, 255, 0),      # 5 green
]


def build_palette_image() -> Image.Image:
    palette = []
    for rgb in SPECTRA_PALETTE:
        palette.extend(rgb)
    palette.extend([255, 255, 255] * (256 - len(SPECTRA_PALETTE)))
    pal = Image.new("P", (1, 1))
    pal.putpalette(palette)
    return pal


def fit_image(image: Image.Image, fit: str, background: tuple[int, int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    target = (LOGICAL_WIDTH, LOGICAL_HEIGHT)

    if fit == "cover":
        return ImageOps.fit(image, target, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    if fit == "stretch":
        return image.resize(target, Image.Resampling.LANCZOS)

    image.thumbnail(target, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target, background)
    x = (LOGICAL_WIDTH - image.width) // 2
    y = (LOGICAL_HEIGHT - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def quantize_to_spectra(image: Image.Image, dither: bool) -> Image.Image:
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return image.quantize(palette=build_palette_image(), dither=dither_mode)


def pack_native_frame(logical: Image.Image) -> bytes:
    """Map the 1600x1200 landscape view to the library's default rotation-1 buffer."""

    pixels = logical.load()
    out = bytearray(DATA_BYTES)

    for y in range(LOGICAL_HEIGHT):
        for x in range(LOGICAL_WIDTH):
            color = pixels[x, y] & 0x0F

            # Inkplate13SPECTRA initDriver() sets rotation 1. writePixel(x, y)
            # maps logical landscape pixels to native framebuffer coordinates:
            # native_col = y, native_row = 1599 - x.
            native_col = y
            native_row = LOGICAL_WIDTH - x - 1
            byte_index = native_row * (NATIVE_WIDTH // 2) + native_col // 2

            if native_col & 1:
                out[byte_index] = (out[byte_index] & 0xF0) | color
            else:
                out[byte_index] = (out[byte_index] & 0x0F) | (color << 4)

    return bytes(out)


def write_bin(output: Path, payload: bytes) -> None:
    flags = 0x03  # native packed data, logical landscape rotation-1 mapping
    header = struct.pack(
        "<8sBBBBHHHHI8s",
        MAGIC,
        VERSION,
        flags,
        len(SPECTRA_PALETTE),
        0,
        LOGICAL_WIDTH,
        LOGICAL_HEIGHT,
        NATIVE_WIDTH,
        NATIVE_HEIGHT,
        len(payload),
        b"\0" * 8,
    )
    output.write_bytes(header + payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image, for example JPG, PNG, or BMP.")
    parser.add_argument("output", type=Path, nargs="?", help="Output .bin path. Defaults beside the input image.")
    parser.add_argument(
        "--fit",
        choices=("cover", "contain", "stretch"),
        default="cover",
        help="Resize mode. cover crops to fill the frame and usually looks best.",
    )
    parser.add_argument(
        "--background",
        default="#FFFFFF",
        help="Background color for --fit contain, for example #FFFFFF.",
    )
    parser.add_argument("--no-dither", action="store_true", help="Disable Floyd-Steinberg palette dithering.")
    parser.add_argument("--preview", type=Path, help="Optional PNG preview of the six-color result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.input.with_suffix(".bin")
    background = ImageColor.getrgb(args.background)

    image = Image.open(args.input)
    fitted = fit_image(image, args.fit, background)
    quantized = quantize_to_spectra(fitted, dither=not args.no_dither)

    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        quantized.convert("RGB").save(args.preview)

    payload = pack_native_frame(quantized)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_bin(output, payload)
    print(f"Wrote {output} ({len(payload) + 32} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
