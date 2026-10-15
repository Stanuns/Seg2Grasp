"""Target-object selection from a set of masks (paper Algorithm 1).

Among all segmented object masks, pick the target ``T`` to grasp: the most
elevated object — i.e. the one whose 3-D centroid is nearest to the (top-down)
camera (smallest depth ``z``). The most elevated object is the most accessible
for suction, and picking it top-down keeps the bin from being disturbed.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TargetObject:
    """The selected target: its index into the mask list, bounding box
    ``[x_min, y_min, x_max, y_max]``, 3-D centroid (mm), and the cropped
    point cloud / mask (and optionally RGB) of just that object's region."""
    index: int
    bbox: np.ndarray
    centroid: np.ndarray
    pc_crop: np.ndarray
    mask_crop: np.ndarray
    rgb_crop: np.ndarray = None


def _mask_centroid_3d(mask, pc_img, patch_frac=0.2, min_points=10):
    """3-D centroid (mm) of a mask, sampled around the mask's pixel centroid.

    Takes the per-axis median of the valid (non-zero depth) points within a small
    patch (radius ``patch_frac * sqrt(area)``) around the pixel centroid. A single
    pixel is unreliable on specular or perforated parts, where depth is often
    missing or wrong; a local median is not, and still tracks the object's centre
    height (e.g. the top of a ball). Falls back to the whole mask if the patch has
    too few valid points. Returns None if the mask has too few valid points.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    cy, cx = ys.mean(), xs.mean()
    r = max(patch_frac * np.sqrt(len(ys)), 3.0)
    near = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
    for sel in (near, slice(None)):
        pts = pc_img[ys[sel], xs[sel]]
        pts = pts[pts[:, 2] > 0]
        if len(pts) >= min_points:
            return np.median(pts, axis=0).astype(np.float32)
    return None


def rank_targets(masks, bboxes, pc_img, rgb_img=None):
    """All objects with a valid 3-D centroid, most elevated (nearest-camera) first.

    Lets the caller fall back to the next object when the most elevated one
    cannot be grasped (e.g. no spot where the suction cup seals).

    Args:
        masks (np.ndarray | list): N boolean masks [H, W].
        bboxes (np.ndarray): N boxes [x_min, y_min, x_max, y_max].
        pc_img (np.ndarray): organized point cloud [H, W, 3] (mm).
        rgb_img (np.ndarray, optional): scene image [H, W, 3] to crop too.

    Returns:
        list of TargetObject, sorted by centroid depth (ascending); may be empty.
    """
    ranked = []
    for idx, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        c = _mask_centroid_3d(mask, pc_img)
        if c is None or np.sum(bboxes[idx]) == 0:
            continue
        x0, y0, x1, y1 = np.round(bboxes[idx]).astype(int)
        ranked.append(TargetObject(
            index=idx,
            bbox=np.array([x0, y0, x1, y1]),
            centroid=c,
            pc_crop=pc_img[y0:y1, x0:x1],
            mask_crop=mask[y0:y1, x0:x1],
            rgb_crop=(rgb_img[y0:y1, x0:x1] if rgb_img is not None else None),
        ))
    ranked.sort(key=lambda t: t.centroid[2])      # smallest depth = most elevated
    return ranked


def select_target(masks, bboxes, pc_img, rgb_img=None):
    """Select the most elevated (nearest-camera) object as the grasp target.

    Returns:
        TargetObject, or None if no mask has a valid 3-D centroid.
        (Arguments as in :func:`rank_targets`.)
    """
    ranked = rank_targets(masks, bboxes, pc_img, rgb_img)
    return ranked[0] if ranked else None
