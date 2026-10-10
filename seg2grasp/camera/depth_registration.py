"""Depth-to-color registration and back-projection (pure NumPy).

Lets Seg2Grasp consume a depth image that is not aligned to the color image
(e.g. the RealSense ROS 2 driver without ``align_depth``): the depth image is
re-projected into the color camera so every color pixel gets a metric depth,
then back-projected into the organized point cloud the pipeline expects.

    depth_c = register_depth_to_color(depth, K_depth, K_color, R, t, rgb.shape[:2])
    pc = backproject(depth_c, K_color)
"""
import numpy as np


def register_depth_to_color(depth, k_depth, k_color, rotation, translation, color_shape):
    """Re-project a depth image into the color camera.

    Each valid depth pixel is back-projected, moved into the color frame and
    splatted over the color pixels its footprint covers (the color camera has a
    narrower field of view, so one depth pixel spans more than one color pixel).
    Where several surfaces land on one color pixel the nearest one wins.

    Args:
        depth (np.ndarray): [Hd, Wd] metric depth (mm) in the depth camera; 0 = invalid.
        k_depth, k_color (np.ndarray): 3x3 pinhole intrinsics (undistorted).
        rotation (np.ndarray): 3x3 depth->color rotation, ``p_c = R @ p_d + t``.
        translation (np.ndarray): depth->color translation (mm).
        color_shape (tuple): (Hc, Wc) of the color image.

    Returns:
        np.ndarray: [Hc, Wc] float32 depth (mm) along the color camera's z axis; 0 = no data.
    """
    hc, wc = color_shape
    kd = np.asarray(k_depth, np.float32)
    kc = np.asarray(k_color, np.float32)
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float32)
    # float32 throughout: int64 pixel indices would promote everything to float64,
    # which also knocks np.minimum.at below off its fast path.
    pts = np.stack([(u.astype(np.float32) - kd[0, 2]) * (z / kd[0, 0]),
                    (v.astype(np.float32) - kd[1, 2]) * (z / kd[1, 1]),
                    z], axis=1)
    pts = pts @ np.asarray(rotation, np.float32).T + np.asarray(translation, np.float32)
    keep = pts[:, 2] > 0
    pts, z = pts[keep], z[keep]
    zc = pts[:, 2]
    uc = pts[:, 0] / zc * kc[0, 0] + kc[0, 2]
    vc = pts[:, 1] / zc * kc[1, 1] + kc[1, 2]

    # Footprint of one depth pixel in the color image, with both ends rounded and
    # inclusive (as librealsense's align does) so neighbouring footprints overlap
    # instead of leaving 1-px seams.
    hx = (kc[0, 0] / kd[0, 0] / 2) * z / zc
    hy = (kc[1, 1] / kd[1, 1] / 2) * z / zc
    u_lo = np.maximum(np.rint(uc - hx), 0).astype(np.int64)
    u_hi = np.minimum(np.rint(uc + hx), wc - 1).astype(np.int64)
    v_lo = np.maximum(np.rint(vc - hy), 0).astype(np.int64)
    v_hi = np.minimum(np.rint(vc + hy), hc - 1).astype(np.int64)
    inside = (u_lo <= u_hi) & (v_lo <= v_hi)
    u_lo, u_hi, v_lo, v_hi, zc = u_lo[inside], u_hi[inside], v_lo[inside], v_hi[inside], zc[inside]

    out = np.full(hc * wc, np.inf, np.float32)
    if len(zc):
        for dy in range(int((v_hi - v_lo).max()) + 1):
            rows = np.flatnonzero(v_lo + dy <= v_hi)
            for dx in range(int((u_hi[rows] - u_lo[rows]).max()) + 1):
                sel = rows[u_lo[rows] + dx <= u_hi[rows]]
                np.minimum.at(out, (v_lo[sel] + dy) * wc + u_lo[sel] + dx, zc[sel])
    out[np.isinf(out)] = 0
    return out.reshape(hc, wc)


def backproject(depth, k):
    """Organized point cloud [H, W, 3] (mm) from a depth image (mm) and its intrinsics."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    return np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], axis=-1)
