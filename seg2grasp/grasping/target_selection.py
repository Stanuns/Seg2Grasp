"""Target-object selection from a set of masks (paper Algorithm 1).

Among all segmented object masks, pick the target ``T`` to grasp: the most
elevated object — i.e. the one whose 3-D centroid is nearest to the (top-down)
camera (smallest depth ``z``). The most elevated object is the most accessible
for suction, and picking it top-down keeps the bin from being disturbed.

Optionally the ranking also weighs each object's segmented area, so a large
workpiece wins over a small object or fragment that is only slightly nearer
(e.g. a part of the robot or clutter at the image border).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TargetObject:
    """The selected target: its index into the mask list, bounding box
    ``[x_min, y_min, x_max, y_max]``, 3-D centroid (mm), the cropped point
    cloud / mask (and optionally RGB) of just that object's region, and its mask
    area (px)."""
    index: int
    bbox: np.ndarray
    centroid: np.ndarray
    pc_crop: np.ndarray
    mask_crop: np.ndarray
    rgb_crop: np.ndarray = None
    area: int = 0


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


def rank_targets(masks, bboxes, pc_img, rgb_img=None, area_weight=0.0, z_range=None):
    """All objects with a valid 3-D centroid, best target first.

    Lets the caller fall back to the next object when the first one cannot be
    grasped (e.g. no spot where the suction cup seals).

    With ``area_weight`` 0 the objects are ranked by elevation alone, most
    elevated (nearest-camera) first. Otherwise by the score
    ``(1 - area_weight) * z_top / z + area_weight * area / area_max``: elevation
    relative to the most elevated object (1 for it, lower the farther an object
    is) blended with the mask area (px) relative to the largest mask. With 1 the
    largest object comes first whatever its height.

    Args:
        masks (np.ndarray | list): N boolean masks [H, W].
        bboxes (np.ndarray): N boxes [x_min, y_min, x_max, y_max].
        pc_img (np.ndarray): organized point cloud [H, W, 3] (mm).
        rgb_img (np.ndarray, optional): scene image [H, W, 3] to crop too.
        area_weight (float): weight of the mask area in the ranking, 0..1.
        z_range (tuple, optional): ``(z_near, z_far)`` (mm); objects whose
            centroid depth is outside it are left out (before the scores are
            normalized, so e.g. the robot's own gripper does not set ``z_top``).

    Returns:
        list of TargetObject, best first; may be empty.
    """
    ranked = []
    for idx, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        c = _mask_centroid_3d(mask, pc_img)
        if c is None or np.sum(bboxes[idx]) == 0:
            continue
        if z_range is not None and not z_range[0] < c[2] < z_range[1]:
            continue
        x0, y0, x1, y1 = np.round(bboxes[idx]).astype(int)
        ranked.append(TargetObject(
            index=idx,
            bbox=np.array([x0, y0, x1, y1]),
            centroid=c,
            pc_crop=pc_img[y0:y1, x0:x1],
            mask_crop=mask[y0:y1, x0:x1],
            rgb_crop=(rgb_img[y0:y1, x0:x1] if rgb_img is not None else None),
            area=int(np.count_nonzero(mask)),
        ))
    if area_weight > 0 and ranked:
        z_top = min(t.centroid[2] for t in ranked)
        area_max = max(t.area for t in ranked)
        ranked.sort(key=lambda t: -((1 - area_weight) * z_top / t.centroid[2] + area_weight * t.area / area_max))
    else:
        ranked.sort(key=lambda t: t.centroid[2])  # smallest depth = most elevated
    return ranked


def select_target(masks, bboxes, pc_img, rgb_img=None):
    """Select the most elevated (nearest-camera) object as the grasp target.

    Returns:
        TargetObject, or None if no mask has a valid 3-D centroid.
        (Arguments as in :func:`rank_targets`.)
    """
    ranked = rank_targets(masks, bboxes, pc_img, rgb_img)
    return ranked[0] if ranked else None
