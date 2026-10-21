"""Through-holes in a part, from the color image.

Instance masks usually cover a part's holes, and the depth camera cannot resolve
small ones (their depth is filled in from the surrounding surface), so neither
tells the suction planner that a cup there would leak. The color image does,
with two cues whose results are combined:

1. Color: through a hole the camera sees the background behind the part.
2. Darkness: seen at an angle, or with little light behind it, a hole shows its
   shaded inner wall and looks much darker than the part around it.

Colors are compared as chromaticity ``(r, g) = (R, G) / (R + G + B)``, which
ignores brightness. Each pixel of the part is projected onto the axis from the
part's typical color (0) to the background's (1); pixels past ``threshold`` are
holes. The axis also covers colors beyond the background: the background seen
through a hole is shaded by the part, which shifts its chromaticity further
along the same direction (e.g. a wooden table turns from yellow to brown).
Only this direction matters: a shiny part reflecting other colors (e.g. bluish
metal) moves away from the background and is never taken for a hole.

A pixel is dark if it is below ``dark_ratio`` times the part's typical
brightness; this is only used on parts that are reasonably bright.

A cue that would flag too much of the part (the part and background colors
cannot be told apart, or the part is unevenly lit) is ignored; with neither cue
the planner falls back to the depth-only seal check.

Main entry point: :func:`detect_holes`.
"""
import cv2
import numpy as np

DARK = 30          # pixels with R + G + B below this have no reliable chromaticity
MIN_BRIGHT = 150   # the darkness cue needs a part at least this bright (median R + G + B)


def _chromaticity(bgr):
    """(r, g) chromaticity [H, W, 2], brightness R + G + B [H, W], and a mask of
    pixels bright enough for a chromaticity."""
    img = bgr.astype(np.float32)
    s = img.sum(axis=2)
    rg = img[:, :, [2, 1]] / np.maximum(s, 1.0)[..., None]
    return rg, s, s >= DARK


def detect_holes(bgr, mask, background, px_per_mm, threshold=0.8, dark_ratio=0.5, max_fraction=0.25,
                 min_diameter=3.0, pad=0.25, min_pixels=500):
    """Pixels of ``mask`` where the background shows through the part.

    Args:
        bgr (np.ndarray): color image [H, W, 3] (BGR).
        mask (np.ndarray): [H, W] bool, the part.
        background (np.ndarray): [H, W] bool, pixels known not to belong to any
            object (e.g. outside all instance masks).
        px_per_mm (float): image scale at the part (focal length in px / depth in mm).
        threshold (float): position on the part (0) -> background (1) color axis
            beyond which a pixel is a hole.
        dark_ratio (float): a pixel darker than this fraction of the part's
            median brightness is a hole (0 disables the darkness cue).
        max_fraction (float): a cue that flags more than this fraction of the
            part is ignored.
        min_diameter (float): holes smaller than this (mm) are ignored as noise.
        pad (float): the background color is taken from the part's bounding box
            enlarged by this fraction on each side.
        min_pixels (int): minimum number of part / background pixels for a model.

    Returns:
        np.ndarray [H, W] bool, the hole pixels (all False if neither cue applies).
    """
    mask = np.asarray(mask, bool)
    holes = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return holes
    # work in the part's bounding box, enlarged by ``pad`` for the background around it
    py, px = int(pad * (ys.max() - ys.min() + 1)), int(pad * (xs.max() - xs.min() + 1))
    win = np.s_[max(ys.min() - py, 0):ys.max() + py + 1, max(xs.min() - px, 0):xs.max() + px + 1]
    rg, brightness, ok = _chromaticity(bgr[win])
    in_part = mask[win]
    if in_part.sum() < min_pixels:
        return holes
    found = np.zeros(in_part.shape, np.uint8)

    # The part's own holes and reflections are a minority of its pixels, so the
    # median gives its typical color and brightness.
    # 1) color: shifted from the part's color toward the background's
    bg = np.asarray(background, bool)[win] & ok
    part = in_part & ok
    if bg.sum() >= min_pixels and part.sum() >= min_pixels:
        mu_part = np.median(rg[part], axis=0)
        axis = np.median(rg[bg], axis=0) - mu_part
        if axis @ axis > 1e-8:
            hole = (rg[part] - mu_part) @ axis / (axis @ axis) > threshold   # 0 = part, 1 = background
            if hole.mean() <= max_fraction:
                found[part] = hole

    # 2) darkness: much darker than the part
    bright = np.median(brightness[in_part])
    if dark_ratio > 0 and bright >= MIN_BRIGHT:
        dark = brightness[in_part] < dark_ratio * bright
        if dark.mean() <= max_fraction:
            found[in_part] |= dark.astype(np.uint8)

    # drop speckle: open, then remove blobs smaller than a min_diameter disc
    found = cv2.morphologyEx(found, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    min_area = np.pi / 4 * (min_diameter * px_per_mm) ** 2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    holes[win] = keep[labels] & mask[win]
    return holes
