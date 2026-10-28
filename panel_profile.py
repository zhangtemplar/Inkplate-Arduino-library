#!/usr/bin/env python3
"""
panel_profile.py -- measure the real primaries of an E Ink Spectra 6 panel.

Subcommands
-----------
  chart    Generate the 6-primary test chart (+ optional flare trap) to show on the panel.
  measure  Sample a photograph of the panel (and of a ColorChecker shot under the same
           light) and emit a JSON profile: XYZ / Lab / sRGB for each of the 6 primaries.
  report   Print a human-readable summary of a profile + a simulated palette preview.

Dependencies
------------
  required : numpy, pillow, matplotlib   (pip install numpy pillow matplotlib)
  optional : rawpy  (strongly recommended -- lets you feed camera RAW files)

Quick start
-----------
  python panel_profile.py chart --width 1200 --height 1600 -o chart.png
  # show chart.png on the panel with dithering DISABLED, then photograph it,
  # and photograph a ColorChecker in the same spot under the same light.

  python panel_profile.py measure \
      --panel panel.ARW --checker checker.ARW \
      -o spectra6_profile.json

  python panel_profile.py report spectra6_profile.json
"""

# ---------------------------------------------------------------- imports
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants

# Nominal palette indices for Spectra 6. The RGB values here are ONLY used to
# paint the test chart; their whole purpose is to be replaced by measurements.
SPECTRA6_NOMINAL = {
    "black":  (0, 0, 0),
    "white":  (255, 255, 255),
    "yellow": (255, 255, 0),
    "red":    (255, 0, 0),
    "blue":   (0, 0, 255),
    "green":  (0, 255, 0),
}
# chart layout order, row-major (3 wide x 2 tall)
CHART_ORDER = ["black", "white", "yellow", "red", "green", "blue"]
CHART_COLS, CHART_ROWS = 3, 2

WP_D65 = np.array([0.95047, 1.00000, 1.08883])
WP_D50 = np.array([0.96422, 1.00000, 0.82521])

XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
SRGB_TO_XYZ = np.linalg.inv(XYZ_TO_SRGB)

BRADFORD = np.array([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
])

# X-Rite ColorChecker Classic, nominal manufacturer Lab under D50.
# >>> These are NOMINAL. Chart-to-chart variation is real (dE ~1-2).
# >>> If you have measured data for YOUR chart, pass it with --checker-ref csv.
CC24_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white 9.5", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black 2",
]
CC24_LAB_D50 = np.array([
    [37.99,  13.56,  14.06], [65.71,  18.13,  17.81], [49.93,  -4.88, -21.93],
    [43.14, -13.10,  21.91], [55.11,   8.84, -25.40], [70.72, -33.40,  -0.20],
    [62.66,  36.07,  57.10], [40.02,  10.41, -45.96], [51.12,  48.24,  16.25],
    [30.33,  22.98, -21.59], [72.53, -23.71,  57.26], [71.94,  19.36,  67.86],
    [28.78,  14.18, -50.30], [55.26, -38.34,  31.37], [42.10,  53.38,  28.19],
    [81.73,   4.04,  79.82], [51.94,  49.99, -14.57], [51.04, -28.63, -28.64],
    [96.54,  -0.43,   1.19], [81.26,  -0.64,  -0.34], [66.77,  -0.73,  -0.50],
    [50.87,  -0.15,  -0.27], [35.66,  -0.42,  -1.23], [20.46,  -0.08,  -0.97],
])
CC_COLS, CC_ROWS = 6, 4

# ---------------------------------------------------------------- color math


def lab_to_xyz(lab, wp=WP_D65):
    lab = np.atleast_2d(np.asarray(lab, float))
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    def finv(t):
        t3 = t ** 3
        return np.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * wp
    return xyz


def xyz_to_lab(xyz, wp=WP_D65):
    xyz = np.atleast_2d(np.asarray(xyz, float))
    r = xyz / wp
    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f(r[:, 0]), f(r[:, 1]), f(r[:, 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def chromatic_adapt(xyz, src_wp, dst_wp):
    """Bradford adaptation."""
    src_lms = BRADFORD @ src_wp
    dst_lms = BRADFORD @ dst_wp
    M = np.linalg.inv(BRADFORD) @ np.diag(dst_lms / src_lms) @ BRADFORD
    return np.atleast_2d(xyz) @ M.T


def xyz_to_srgb_linear(xyz):
    return np.atleast_2d(xyz) @ XYZ_TO_SRGB.T


def srgb_encode(lin):
    lin = np.clip(np.asarray(lin, float), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def srgb_decode(enc):
    enc = np.asarray(enc, float)
    return np.where(enc <= 0.04045, enc / 12.92, ((enc + 0.055) / 1.055) ** 2.4)


def delta_e_2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    lab1 = np.atleast_2d(np.asarray(lab1, float))
    lab2 = np.atleast_2d(np.asarray(lab2, float))
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7 + 1e-30)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))
    Lbar = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbarp = np.where(
        C1p * C2p == 0, hsum,
        np.where(np.abs(h1p - h2p) <= 180, hsum / 2.0,
                 np.where(hsum < 360, (hsum + 360) / 2.0, (hsum - 360) / 2.0)))
    T = (1 - 0.17 * np.cos(np.radians(hbarp - 30))
         + 0.24 * np.cos(np.radians(2 * hbarp))
         + 0.32 * np.cos(np.radians(3 * hbarp + 6))
         - 0.20 * np.cos(np.radians(4 * hbarp - 63)))
    dtheta = 30 * np.exp(-(((hbarp - 275) / 25.0) ** 2))
    Rc = 2 * np.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1 + (0.015 * (Lbar - 50) ** 2) / np.sqrt(20 + (Lbar - 50) ** 2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
                   + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))


def cc24_xyz_d65(lab_d50=None):
    lab = CC24_LAB_D50 if lab_d50 is None else np.asarray(lab_d50, float)
    xyz50 = lab_to_xyz(lab, WP_D50)
    return chromatic_adapt(xyz50, WP_D50, WP_D65)


# ---------------------------------------------------------------- chart


def cmd_chart(args):
    from PIL import Image, ImageDraw

    W, H = args.width, args.height
    img = Image.new("RGB", (W, H), SPECTRA6_NOMINAL["white"])
    d = ImageDraw.Draw(img)

    margin = int(min(W, H) * 0.06)
    gap = int(min(W, H) * 0.02)
    grid_w = W - 2 * margin
    grid_h = int((H - 2 * margin) * (0.78 if args.flare_trap else 1.0))
    cw = (grid_w - (CHART_COLS - 1) * gap) // CHART_COLS
    ch = (grid_h - (CHART_ROWS - 1) * gap) // CHART_ROWS

    cells = []
    for i, name in enumerate(CHART_ORDER):
        r, c = divmod(i, CHART_COLS)
        x0 = margin + c * (cw + gap)
        y0 = margin + r * (ch + gap)
        d.rectangle([x0, y0, x0 + cw, y0 + ch], fill=SPECTRA6_NOMINAL[name])
        cells.append({"name": name, "box": [x0, y0, x0 + cw, y0 + ch]})

    # Registration marks: black L-corners just outside the grid, so you can click
    # the 4 grid corners accurately in the picker.
    t = max(3, min(W, H) // 300)
    gx0, gy0 = margin, margin
    gx1, gy1 = margin + CHART_COLS * cw + (CHART_COLS - 1) * gap, margin + grid_h
    arm = int(min(W, H) * 0.03)
    for (px, py, sx, sy) in [(gx0, gy0, 1, 1), (gx1, gy0, -1, 1),
                             (gx0, gy1, 1, -1), (gx1, gy1, -1, -1)]:
        d.line([px, py, px + sx * arm, py], fill=(0, 0, 0), width=t)
        d.line([px, py, px, py + sy * arm], fill=(0, 0, 0), width=t)

    if args.flare_trap:
        # A large black field at the bottom. Photograph it to estimate veiling
        # glare / camera black offset. Even better: cut a hole in black card and
        # shoot into a dark box -- but this is a decent cheap proxy.
        ty0 = gy1 + gap * 2
        d.rectangle([margin, ty0, gx1, H - margin], fill=(0, 0, 0))
        cells.append({"name": "flare_trap", "box": [margin, ty0, gx1, H - margin]})

    out = Path(args.output)
    img.save(out)
    meta = {"width": W, "height": H, "order": CHART_ORDER,
            "cols": CHART_COLS, "rows": CHART_ROWS, "cells": cells}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")
    print("IMPORTANT: display this with dithering/colour-management DISABLED, so each")
    print("patch maps 1:1 to a single panel primary. Use a full refresh at room temp.")


# ---------------------------------------------------------------- image loading


def load_linear_rgb(path, raw_wb="none", half_size=False):
    """Return float32 HxWx3 *scene-linear* RGB, plus a tag describing the space."""
    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im).astype(np.float32)
        maxv = 65535.0 if arr.max() > 255.0 else 255.0
        arr /= maxv
        if path.suffix.lower() in {".tif", ".tiff"} and maxv == 65535.0:
            # assume 16-bit TIFF from a raw converter is already linear
            return arr, "linear-unknown(16bit tiff)"
        return srgb_decode(arr).astype(np.float32), "linear-sRGB(from 8bit, LOSSY)"

    try:
        import rawpy
    except ImportError:
        raise SystemExit("rawpy is required for RAW files:  pip install rawpy")
    with rawpy.imread(str(path)) as raw:
        kw = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                  output_color=rawpy.ColorSpace.raw, half_size=half_size,
                  user_flip=0, highlight_mode=rawpy.HighlightMode.Clip)
        if raw_wb == "camera":
            kw["use_camera_wb"] = True
        else:
            kw["use_camera_wb"] = False
            kw["use_auto_wb"] = False
            kw["user_wb"] = [1.0, 1.0, 1.0, 1.0]
        rgb = raw.postprocess(**kw).astype(np.float32) / 65535.0
    return rgb, "linear-camera-native"


# ---------------------------------------------------------------- patch sampling


def grid_centers(corners, cols, rows):
    """corners = [TL, TR, BR, BL] in pixel coords -> (rows*cols, 2) centers."""
    tl, tr, br, bl = [np.asarray(c, float) for c in corners]
    pts = []
    for r in range(rows):
        v = (r + 0.5) / rows
        for c in range(cols):
            u = (c + 0.5) / cols
            top = tl + (tr - tl) * u
            bot = bl + (br - bl) * u
            pts.append(top + (bot - top) * v)
    return np.array(pts)


def sample_patches(img, centers, cols, rows, corners, frac=0.35):
    """Median-sample a square at each center. Returns (N,3) plus clipping flags."""
    tl, tr, br, bl = [np.asarray(c, float) for c in corners]
    cell_w = np.linalg.norm(tr - tl) / cols
    cell_h = np.linalg.norm(bl - tl) / rows
    half = max(2, int(min(cell_w, cell_h) * frac / 2))
    H, W = img.shape[:2]
    vals, clipped, noisy = [], [], []
    for (x, y) in centers:
        x0, x1 = int(max(0, x - half)), int(min(W, x + half))
        y0, y1 = int(max(0, y - half)), int(min(H, y + half))
        blk = img[y0:y1, x0:x1].reshape(-1, 3)
        vals.append(np.median(blk, axis=0))
        clipped.append(bool((blk > 0.995).any()))
        # relative spread, useful for spotting glare streaks / mis-clicks
        m = np.median(blk, axis=0)
        noisy.append(float(np.median(np.abs(blk - m)) / (m.mean() + 1e-6)))
    return np.array(vals), clipped, noisy


def pick_corners(img, title, preview_gamma=True):
    import matplotlib
    import matplotlib.pyplot as plt
    disp = srgb_encode(img / max(1e-6, np.percentile(img, 99.5))) if preview_gamma else img
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(np.clip(disp, 0, 1))
    ax.set_title(f"{title}\nclick 4 corners: TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT"
                 "\n(centres of the corner patches; right-click undo, middle-click done)")
    ax.axis("off")
    pts = plt.ginput(4, timeout=0)
    plt.close(fig)
    if len(pts) != 4:
        raise SystemExit("need exactly 4 corner clicks")
    return pts


# ---------------------------------------------------------------- matrix solve


def solve_cam_to_xyz(cam_rgb, xyz_ref, mode="rootpoly", weights=None):
    """Least-squares camera-linear -> XYZ(D65).

    mode 'linear'   : 3x3
    mode 'rootpoly' : root-polynomial degree 2 (Finlayson et al.) -- exposure
                      invariant and clearly more accurate than a bare 3x3.
    Returns (apply_fn, residual_dE_array).
    """
    def feats(rgb):
        R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        if mode == "linear":
            return np.stack([R, G, B], axis=1)
        eps = 1e-12
        return np.stack([
            R, G, B,
            np.sqrt(np.abs(R * G) + eps), np.sqrt(np.abs(G * B) + eps),
            np.sqrt(np.abs(R * B) + eps),
        ], axis=1)

    A = feats(cam_rgb)
    W = np.ones(len(A)) if weights is None else np.asarray(weights, float)
    Aw = A * W[:, None]
    Bw = xyz_ref * W[:, None]
    M, *_ = np.linalg.lstsq(Aw, Bw, rcond=None)

    def apply(rgb):
        return feats(np.atleast_2d(np.asarray(rgb, float))) @ M

    res = delta_e_2000(xyz_to_lab(apply(cam_rgb)), xyz_to_lab(xyz_ref))
    return apply, res


# ---------------------------------------------------------------- measure


def cmd_measure(args):
    panel_img, panel_space = load_linear_rgb(args.panel, raw_wb="none",
                                             half_size=args.half_size)
    print(f"panel image : {panel_img.shape} {panel_space}")

    # ---- flare / black offset -------------------------------------------
    flare = np.zeros(3)
    if args.flare is not None:
        flare = np.array([float(v) for v in args.flare.split(",")])
        print(f"flare offset (manual): {flare}")

    # ---- camera -> XYZ ---------------------------------------------------
    if args.checker:
        chk_img, chk_space = load_linear_rgb(args.checker, raw_wb="none",
                                             half_size=args.half_size)
        print(f"checker image: {chk_img.shape} {chk_space}")
        if args.checker_corners:
            corners = np.array(json.loads(args.checker_corners), float)
        else:
            corners = pick_corners(chk_img, "ColorChecker (6 wide x 4 tall)")
        centers = grid_centers(corners, CC_COLS, CC_ROWS)
        cam, clipped, noisy = sample_patches(chk_img, centers, CC_COLS, CC_ROWS, corners)
        if any(clipped):
            bad = [CC24_NAMES[i] for i, c in enumerate(clipped) if c]
            print(f"  WARNING clipped checker patches: {bad} -> reduce exposure and reshoot")
        ref_lab = CC24_LAB_D50
        if args.checker_ref:
            ref_lab = np.loadtxt(args.checker_ref, delimiter=",")[:, :3]
            print(f"  using measured checker reference from {args.checker_ref}")
        xyz_ref = cc24_xyz_d65(ref_lab)

        cam_c = np.clip(cam - flare, 1e-7, None)
        # down-weight the darkest patch (most flare-contaminated) a little
        w = np.ones(24); w[23] = 0.5
        apply_cam, res = solve_cam_to_xyz(cam_c, xyz_ref, mode=args.fit, weights=w)
        print(f"  fit={args.fit}  checker residual dE2000: "
              f"mean {res.mean():.2f}  max {res.max():.2f} ({CC24_NAMES[int(res.argmax())]})")
        if res.mean() > 3.0:
            print("  WARNING mean dE>3 -- suspect mixed lighting, wrong patch clicks,")
            print("          clipping, or an 8-bit JPEG input.")
        cc_debug = {"cam_linear": cam.tolist(), "residual_dE2000": res.tolist(),
                    "corners": np.asarray(corners).tolist()}
    else:
        print("NO --checker given: falling back to 'camera == sRGB primaries' + grey-card")
        print("white balance. Expect several dE of systematic error, especially in")
        print("saturated reds/blues. Shoot a ColorChecker if you can.")
        if not args.white_ref:
            raise SystemExit("without --checker you must give --white-ref x,y and --white-L")
        wx, wy = [int(v) for v in args.white_ref.split(",")]
        patch = panel_img[wy - 20:wy + 20, wx - 20:wx + 20].reshape(-1, 3)
        wb = np.median(patch, axis=0)
        Y_target = lab_to_xyz([[args.white_L, 0, 0]], WP_D65)[0, 1]
        gain = (Y_target / max(wb.mean(), 1e-6))

        def apply_cam(rgb):
            rgb = np.atleast_2d(np.asarray(rgb, float))
            bal = rgb / wb * wb.mean() * gain
            return xyz_to_srgb_linear(bal) if False else bal @ SRGB_TO_XYZ.T
        cc_debug = None

    # ---- panel patches ----------------------------------------------------
    if args.panel_corners:
        pcorners = np.array(json.loads(args.panel_corners), float)
    else:
        pcorners = pick_corners(panel_img, "panel chart (3 wide x 2 tall)")
    pcent = grid_centers(pcorners, CHART_COLS, CHART_ROWS)
    pcam, pclip, pnoisy = sample_patches(panel_img, pcent, CHART_COLS, CHART_ROWS, pcorners)
    if any(pclip):
        bad = [CHART_ORDER[i] for i, c in enumerate(pclip) if c]
        print(f"  WARNING clipped panel patches: {bad} -> reduce exposure and reshoot")
    for n, s in zip(CHART_ORDER, pnoisy):
        if s > 0.12:
            print(f"  NOTE patch '{n}' is non-uniform (spread {s:.2f}) -- glare or ghosting?")

    pcam_c = np.clip(pcam - flare, 1e-7, None)
    xyz = apply_cam(pcam_c)

    # Normalise so a perfect diffuse reflector under this light would be Y=1.
    # The checker fit already puts us on that scale, so only optionally rescale
    # to a user-asserted panel white L*.
    if args.force_white_L is not None:
        wi = CHART_ORDER.index("white")
        target_Y = lab_to_xyz([[args.force_white_L, 0, 0]], WP_D65)[0, 1]
        xyz = xyz * (target_Y / max(xyz[wi, 1], 1e-9))
        print(f"  rescaled so panel white L* = {args.force_white_L}")

    lab = xyz_to_lab(xyz)
    lin = xyz_to_srgb_linear(xyz)
    srgb8 = np.round(np.clip(srgb_encode(lin), 0, 1) * 255).astype(int)

    primaries = []
    for i, name in enumerate(CHART_ORDER):
        primaries.append({
            "name": name,
            "index": i,
            "nominal_srgb8": list(SPECTRA6_NOMINAL[name]),
            "XYZ_D65": [round(float(v), 6) for v in xyz[i]],
            "Lab_D65": [round(float(v), 3) for v in lab[i]],
            "srgb_linear": [round(float(v), 6) for v in lin[i]],
            "srgb8": [int(v) for v in srgb8[i]],
            "srgb8_clipped": bool((lin[i] < -0.002).any() or (lin[i] > 1.002).any()),
            "camera_linear_raw": [round(float(v), 6) for v in pcam[i]],
        })

    profile = {
        "schema": "spectra6-panel-profile/1",
        "illuminant": "D65 (assumed; via ColorChecker fit)" if args.checker else "grey-card WB",
        "fit": args.fit if args.checker else "diagonal-wb",
        "panel_image": str(args.panel),
        "checker_image": str(args.checker) if args.checker else None,
        "flare_offset_camera_linear": [float(v) for v in flare],
        "notes": args.notes,
        "primaries": primaries,
        "_debug": {"panel_corners": np.asarray(pcorners).tolist(), "checker": cc_debug},
    }
    Path(args.output).write_text(json.dumps(profile, indent=2))
    print(f"wrote {args.output}")
    report_profile(profile)
    return profile


# ---------------------------------------------------------------- report


def report_profile(profile):
    prim = profile["primaries"]
    lab = np.array([p["Lab_D65"] for p in prim])
    print()
    print(f"{'name':8s} {'L*':>7s} {'a*':>7s} {'b*':>7s} {'C*':>7s} {'h':>6s}  sRGB8")
    for p, l in zip(prim, lab):
        C = float(np.hypot(l[1], l[2]))
        h = float(np.degrees(np.arctan2(l[2], l[1])) % 360)
        flag = " (out of sRGB)" if p["srgb8_clipped"] else ""
        print(f"{p['name']:8s} {l[0]:7.2f} {l[1]:7.2f} {l[2]:7.2f} {C:7.2f} {h:6.1f}  "
              f"{tuple(p['srgb8'])}{flag}")
    names = [p["name"] for p in prim]
    Lw = lab[names.index("white"), 0]
    Lk = lab[names.index("black"), 0]
    Yw = prim[names.index("white")]["XYZ_D65"][1]
    Yk = prim[names.index("black")]["XYZ_D65"][1]
    print()
    print(f"dynamic range : L* {Lk:.1f} .. {Lw:.1f}   (contrast ratio {Yw / max(Yk,1e-6):.1f}:1)")
    print(f"max chroma    : C* {np.hypot(lab[:,1], lab[:,2]).max():.1f}")
    d = delta_e_2000(np.repeat(lab, len(lab), axis=0), np.tile(lab, (len(lab), 1)))
    d = d.reshape(len(lab), len(lab)) + np.eye(len(lab)) * 999
    i, j = np.unravel_index(d.argmin(), d.shape)
    print(f"closest pair  : {names[i]} / {names[j]}  dE2000 = {d[i,j]:.1f}"
          "   (small => they will be confused by the quantiser)")


def cmd_report(args):
    profile = json.loads(Path(args.profile).read_text())
    report_profile(profile)
    if args.swatch:
        from PIL import Image
        prim = profile["primaries"]
        sz = 120
        img = Image.new("RGB", (sz * len(prim), sz * 2), (128, 128, 128))
        for i, p in enumerate(prim):
            img.paste(tuple(p["nominal_srgb8"]), (i * sz, 0, (i + 1) * sz, sz))
            img.paste(tuple(p["srgb8"]), (i * sz, sz, (i + 1) * sz, sz * 2))
        img.save(args.swatch)
        print(f"wrote {args.swatch}  (top row = nominal, bottom row = measured)")


# ---------------------------------------------------------------- cli


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chart", help="generate the panel test chart")
    c.add_argument("--width", type=int, default=1200)
    c.add_argument("--height", type=int, default=1600)
    c.add_argument("-o", "--output", default="chart.png")
    c.add_argument("--flare-trap", action="store_true", default=True)
    c.add_argument("--no-flare-trap", dest="flare_trap", action="store_false")
    c.set_defaults(func=cmd_chart)

    m = sub.add_parser("measure", help="sample photos and emit a profile")
    m.add_argument("--panel", required=True, help="photo of the panel showing chart.png")
    m.add_argument("--checker", help="photo of a ColorChecker under the SAME light")
    m.add_argument("--checker-ref", help="CSV of measured Lab(D50) for your chart, 24 rows")
    m.add_argument("--fit", choices=["linear", "rootpoly"], default="rootpoly")
    m.add_argument("--flare", help="camera-linear RGB to subtract, e.g. 0.004,0.004,0.005")
    m.add_argument("--panel-corners", help="JSON [[x,y]x4] to skip the clicking UI")
    m.add_argument("--checker-corners", help="JSON [[x,y]x4]")
    m.add_argument("--white-ref", help="x,y pixel of a grey card (no-checker fallback)")
    m.add_argument("--white-L", type=float, default=96.0, help="L* of that grey card")
    m.add_argument("--force-white-L", type=float, default=None,
                   help="rescale so panel white hits this L* (use only if you trust it)")
    m.add_argument("--half-size", action="store_true", help="faster RAW decode")
    m.add_argument("--notes", default="")
    m.add_argument("-o", "--output", default="spectra6_profile.json")
    m.set_defaults(func=cmd_measure)

    r = sub.add_parser("report", help="summarise a profile")
    r.add_argument("profile")
    r.add_argument("--swatch", help="write a nominal-vs-measured swatch PNG here")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
