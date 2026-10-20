"""Analytic suction-point estimation (paper Algorithm 2).

Given the point cloud of a target object, find the optimal suction pose
``(point, normal)``: a locally flat, cup-sized patch whose surface normal faces
the camera. Non-learning (no trained weights): geometrically-uniform candidate
points are evaluated over the object surface and the best is chosen by a weighted
score (paper Eq. 2) of flatness, proximity to the middle of the object, and
graspable-point count. A RANSAC cup-disc search is kept as a fallback.

Main entry point: :func:`estimate_suction_point`.
"""
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SuctionPose:
    """A suction grasp: 3-D contact ``point`` (mm) and unit surface ``normal``
    (pointing toward the camera, i.e. ``normal[2] < 0``)."""
    point: np.ndarray
    normal: np.ndarray


def fit_plane_lstsq(pts):
    """Least-squares plane fit through 3-D points.

    Returns the plane as ``[a, b, c, d]`` with unit normal ``[a, b, c]`` such
    that ``a*x + b*y + c*z + d = 0``.
    """
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    A = np.c_[xs, ys, np.ones(len(xs))]
    b = zs.reshape(-1, 1)
    fit = np.linalg.lstsq(A, b, rcond=None)[0].ravel()  # z = fit0*x + fit1*y + fit2
    vector = np.array([-fit[0], -fit[1], 1.0])
    normal = vector / np.linalg.norm(vector)
    plane = np.concatenate((normal, [-fit[2] * normal[2]]))
    return plane


def remove_background(pc_img, z_near=300.0, z_far=1050.0):
    """Keep only object-surface points within a metric depth band (mm)."""
    pts = pc_img[(pc_img[:, :, 2] < z_far) & (pc_img[:, :, 2] > z_near)]
    return pts.reshape(-1, 3)


def _voxel_downsample(points, voxel):
    """One representative point per voxel — gives spatially-uniform seeds."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def suction_candidates(surface, vacuum_radius=30.0, spacing=None, min_points=8, flat_tol=3.0,
                       margin=1.5, extra_seeds=None):
    """Evaluate geometrically-uniform candidate suction points over a surface.

    Seeds are placed uniformly (one per ``spacing``-sized voxel, default = 2/3 of
    the cup radius, so the best spot on the object is never far from a seed), plus
    any ``extra_seeds`` (e.g. the object centre). Each seed gathers the points within
    ``vacuum_radius`` and is a *valid* candidate if it has enough points and the
    patch is flat enough (mean abs. distance to its fitted plane ≤ ``flat_tol`` mm).

    ``margin_dev`` measures how flat the surroundings are: the 90th-percentile
    distance to the patch plane of the points within ``margin * vacuum_radius``.
    A patch on, or right next to, a bend, rib or crease has a flat-looking core
    but a large ``margin_dev``, because the surface leaves the plane around it.

    Returns a list of dicts per candidate: ``{center, point, normal, flat,
    margin_dev, n, valid}`` where ``point`` is the patch centre of gravity (mean
    of its points) and ``normal`` is that patch's camera-facing unit normal.
    """
    spacing = spacing or vacuum_radius * 2 / 3
    seeds = _voxel_downsample(surface, spacing)
    if extra_seeds is not None and len(extra_seeds):
        seeds = np.vstack([np.asarray(extra_seeds, seeds.dtype).reshape(-1, 3), seeds])
    n_seeds = len(seeds)
    if n_seeds == 0:
        return []

    # All seeds at once (vectorised; a per-seed Python loop dominated the frame time
    # for small cups). Flatten every seed's neighbourhood into one array, tagged
    # with its seed id, and reduce per seed with bincount.
    neighbourhoods = cKDTree(surface).query_ball_point(seeds, margin * vacuum_radius)
    lens = np.fromiter(map(len, neighbourhoods), np.int64, n_seeds)
    gid = np.repeat(np.arange(n_seeds), lens)
    pts = surface[np.fromiter(itertools.chain.from_iterable(neighbourhoods), np.int64, lens.sum())]
    pts = pts.astype(np.float64)
    dist = np.linalg.norm(pts - seeds[gid], axis=1)
    keep = dist < margin * vacuum_radius
    gid, pts, dist = gid[keep], pts[keep], dist[keep]
    near = dist < vacuum_radius
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    def per_seed(w, sel=near):
        return np.bincount(gid[sel], weights=w[sel], minlength=n_seeds)

    n = np.bincount(gid[near], minlength=n_seeds).astype(np.float64)
    sx, sy, sz = per_seed(x), per_seed(y), per_seed(z)
    # least-squares plane z = a*x + b*y + c per seed via the normal equations
    # (pinv = lstsq's minimum-norm solution, also for degenerate patches)
    A = np.stack([np.stack([per_seed(x * x), per_seed(x * y), sx], -1),
                  np.stack([per_seed(x * y), per_seed(y * y), sy], -1),
                  np.stack([sx, sy, n], -1)], axis=1)
    rhs = np.stack([per_seed(x * z), per_seed(y * z), sz], -1)
    ok = n >= min_points
    coef = np.zeros((n_seeds, 3))
    if ok.any():
        coef[ok] = (np.linalg.pinv(A[ok]) @ rhs[ok][..., None])[..., 0]
    a, b, c = coef[:, 0], coef[:, 1], coef[:, 2]
    norm = np.sqrt(a * a + b * b + 1.0)
    res = np.abs(z - (a[gid] * x + b[gid] * y + c[gid])) / norm[gid]   # point-to-plane distance

    flat = per_seed(res) / np.maximum(n, 1)
    # 90th percentile of the residuals within the margin, per seed (np.percentile's
    # linear interpolation on each seed's sorted residuals). gid is already sorted,
    # so offsetting each seed's residuals by gid * span sorts them per seed in one
    # plain sort (several times faster than lexsort).
    span = res.max() + 1.0 if len(res) else 1.0
    res_sorted = np.sort(gid * span + res) - gid * span
    counts = np.bincount(gid, minlength=n_seeds)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pos = 0.9 * np.maximum(counts - 1, 0)
    lo, hi = np.floor(pos).astype(np.int64), np.ceil(pos).astype(np.int64)
    take = lambda k: res_sorted[np.minimum(starts + k, len(res_sorted) - 1)]
    margin_dev = take(lo) + (take(hi) - take(lo)) * (pos - lo)

    centre = np.stack([sx, sy, sz], -1) / np.maximum(n, 1)[:, None]
    normal = np.stack([a, b, -np.ones_like(a)], -1) / norm[:, None]   # camera-facing (z < 0)
    return [dict(center=seeds[i], point=centre[i], normal=normal[i], flat=float(flat[i]),
                 margin_dev=float(margin_dev[i]), n=int(n[i]), valid=bool(flat[i] <= flat_tol))
            for i in np.flatnonzero(ok)]


def _pixel_model(pc_img, max_samples=5000):
    """Recover the pinhole mapping ``u = fx*x/z + cx, v = fy*y/z + cy`` of an
    organized cloud, so pixels without depth can still be cast as rays. Works on
    crops (the crop offset folds into ``cx, cy``). None if it cannot be fitted."""
    v, u = np.nonzero(pc_img[:, :, 2] > 0)
    if len(u) < 50:
        return None
    step = max(len(u) // max_samples, 1)
    u, v = u[::step], v[::step]
    p = pc_img[v, u].astype(np.float64)
    a, b = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    if np.ptp(a) < 1e-6 or np.ptp(b) < 1e-6:
        return None
    fx, cx = np.polyfit(a, u, 1)
    fy, cy = np.polyfit(b, v, 1)
    return fx, cx, fy, cy


def seal_ratio(pc_img, member, model, point, normal, vacuum_radius=30.0, tol=5.0,
               holes=None, hole_margin=2.0):
    """Fraction of the suction-cup disc that can seal.

    The cup is a disc of ``vacuum_radius`` centred on ``point`` in the plane with
    ``normal``. Every pixel whose camera ray hits that disc is checked: it seals
    only if it is on the object (``member``), has valid depth, and its measured
    point lies within ``tol`` mm of the plane. Holes (no depth, or depth seen
    through the part), edges (off the object) and steps or creases (off the
    plane) all lower the ratio. A hole known from ``holes`` (e.g. from the color
    image, see :mod:`.hole_detection`) within ``vacuum_radius + hole_margin``
    makes the cup leak however small the overlap, so the ratio is then 0.

    Args:
        pc_img (np.ndarray): organized point cloud [H, W, 3] (mm).
        member (np.ndarray): [H, W] bool, pixels belonging to the object.
        model (tuple): ``(fx, cx, fy, cy)`` from :func:`_pixel_model`.
        point, normal (np.ndarray): cup centre (mm) and unit plane normal.
        holes (np.ndarray, optional): [H, W] bool, known hole pixels.
        hole_margin (float): solid material (mm) required between the cup and a hole.

    Returns:
        float in [0, 1].
    """
    fx, cx, fy, cy = model
    h, w = member.shape
    point, normal = np.asarray(point, np.float64), np.asarray(normal, np.float64)
    z = point[2]
    u0, v0 = fx * point[0] / z + cx, fy * point[1] / z + cy
    reach = vacuum_radius + (hole_margin if holes is not None else 0.0)
    # a tilted disc projects inside this window (its nearest part is at depth >= z - r)
    zmin = max(z - reach, 1.0)
    ru, rv = reach * abs(fx) / zmin + 1, reach * abs(fy) / zmin + 1
    uu, vv = np.meshgrid(np.arange(np.floor(u0 - ru), np.ceil(u0 + ru) + 1, dtype=np.int64),
                         np.arange(np.floor(v0 - rv), np.ceil(v0 + rv) + 1, dtype=np.int64))
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], axis=-1)
    denom = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (point @ normal) / denom                      # ray-plane intersection
    hit = rays * t[..., None]
    radial = np.where((np.abs(denom) > 1e-9) & (t > 0), np.linalg.norm(hit - point, axis=-1), np.inf)
    in_disc = radial <= vacuum_radius
    if not in_disc.any():
        return 0.0
    in_image = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    if holes is not None:
        near = (radial <= reach) & in_image
        if holes[vv[near], uu[near]].any():
            return 0.0
    inside = in_disc & in_image
    ui, vi = uu[inside], vv[inside]
    pts = pc_img[vi, ui]
    good = member[vi, ui] & (pts[:, 2] > 0) & (np.abs((pts - point) @ normal) <= tol)
    return float(good.sum() / in_disc.sum())             # disc pixels outside the crop count as unsealed


def _object_centre(pc_img, region, member, model, surface):
    """Centre of the object, for the centrality score and an extra seed.

    In the image when the pixel model is known: the centroid of the object's mask
    pixels — including pixels without depth, so missing depth on shiny or
    perforated parts does not drag it — with the object's equivalent radius
    ``sqrt(area / pi)`` (px) as the scale. Otherwise in 3-D: the surface centroid
    and the radius of a disc with the surface's spread.

    Returns ``(centre, scale, seed)``: ``centre`` is ``(u, v)`` px in the image or
    a 3-D point (mm), and ``seed`` is the 3-D surface point nearest the centre.
    """
    if model is not None:
        vs, us = np.nonzero(region)
        centre = np.array([us.mean(), vs.mean()])
        scale = max(np.sqrt(len(us) / np.pi), 1.0)
        mv, mu = np.nonzero(member)
        k = np.argmin((mu - centre[0]) ** 2 + (mv - centre[1]) ** 2)
        return centre, scale, pc_img[mv[k], mu[k]]
    centre = surface.mean(axis=0)
    # disc of radius R: mean squared distance from its centre is R^2 / 2
    scale = max(np.sqrt(2 * np.mean(np.sum((surface - centre) ** 2, axis=1))), 1.0)
    return centre, scale, surface[np.argmin(np.linalg.norm(surface - centre, axis=1))]


def centre_distance(points, centre, scale, model=None):
    """Distance of ``points`` (N x 3, mm) from the object centre, in units of the
    object's equivalent radius (0 = centre, about 1 = edge); measured in the image
    if ``model`` is given (see :func:`_object_centre`)."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    if model is not None:
        fx, cx, fy, cy = model
        z = np.maximum(points[:, 2], 1e-6)
        uv = np.stack([fx * points[:, 0] / z + cx, fy * points[:, 1] / z + cy], -1)
        return np.linalg.norm(uv - centre, axis=1) / scale
    return np.linalg.norm(points - centre, axis=1) / scale


def score_candidates(candidates, centroid=None, w_angle=0.25, w_dist=0.3, w_count=0.05,
                     w_flat=0.4, flat_tol=3.0):
    """Score valid suction candidates (paper Eq. 2, plus a flatness term): weighted
    sum of the surface angle (how top-facing the patch normal is), proximity to
    the middle of the object, graspable-point count, and flatness of the patch's
    surroundings. Weights sum to 1. Fills ``candidate['score']`` and returns the
    valid list.

    The centrality term is ``1 - centre_dist`` (clipped to [0, 1]), where
    ``candidate['centre_dist']`` is the distance from the object centre in units of
    the object's equivalent radius (see :func:`centre_distance`); without it, the
    3-D distance to ``centroid`` relative to the farthest candidate is used. It is
    an absolute measure, so a candidate halfway to the edge loses half the term
    whatever the other candidates are: on a flat face, where the other terms tie,
    the cup goes to the middle; a bend, rib or crease there still outweighs it.

    The angle term is ``|normal_z|`` — the alignment of the patch normal with the
    camera/approach axis — so a top-facing patch (normal pointing straight up at
    the camera) scores highest and a steep side patch on a round object scores
    low. The flatness term is ``1 - (margin_dev - floor) / flat_tol`` (clipped to
    [0, 1]), where ``floor`` (10th percentile of ``margin_dev`` over the candidates)
    is the object's own sensor-noise level: it steers the cup away from bends,
    ribs and creases, but sensor noise alone does not move it off the middle of a
    flat face.
    """
    valid = [c for c in candidates if c["valid"]]
    if not valid:
        return []
    counts = np.array([c["n"] for c in valid], dtype=float)
    s_angle = np.array([abs(c["normal"][2]) for c in valid])   # 1 = top-facing, 0 = vertical wall
    if all("centre_dist" in c for c in valid):
        s_dist = np.clip(1 - np.array([c["centre_dist"] for c in valid]), 0, 1)  # 1 = middle
    else:
        dists = np.array([np.linalg.norm(c["point"] - centroid) for c in valid])
        s_dist = 1 - dists / (dists.max() + 1e-9)               # closer to CoG -> more stable
    s_count = counts / (counts.max() + 1e-9)                    # more support -> better
    margin_dev = np.array([c["margin_dev"] for c in valid])
    s_flat = np.clip(1 - (margin_dev - np.percentile(margin_dev, 10)) / flat_tol, 0, 1)  # 1 = flat surroundings
    scores = w_angle * s_angle + w_dist * s_dist + w_count * s_count + w_flat * s_flat
    for c, sc in zip(valid, scores):
        c["score"] = float(sc)
    return valid


def ransac_disc_fit(surface, vacuum_radius=30.0, threshold=3.0, n_iter=500, rng=None):
    """Search for the best cup-sized planar disc on the object surface.

    RANSAC: repeatedly fit a plane to 3 random points, take its inliers, then
    look for the densest ``vacuum_radius``-sized disc of inliers (a patch the
    suction cup can seal against). Returns the plane, the disc's inlier indices
    (into ``surface``), the plane's full inlier surface, and the iteration count.
    """
    rng = rng or np.random.default_rng()
    n_points = len(surface)
    best_disc_idx = np.array([0])
    best_plane = np.array([0.0, 0.0, 1.0, 0.0])
    best_plane_surface = None
    last_hit = 0

    for i in range(n_iter):
        try:
            sample = surface[rng.choice(n_points, 3, replace=False)]
            a, b, c, d = fit_plane_lstsq(sample)
        except Exception:
            continue

        dist = (a * surface[:, 0] + b * surface[:, 1] + c * surface[:, 2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)
        plane_idx = np.where(np.abs(dist) <= threshold)[0]
        if len(plane_idx) <= len(best_disc_idx):
            continue

        plane_surface = surface[plane_idx]
        for j in range(len(plane_surface) // 2 - 1):
            seed = np.tile(plane_surface[j * 2], (plane_surface.shape[0], 1))
            disc_idx = np.where(np.linalg.norm(seed - plane_surface, axis=-1) < vacuum_radius)[0]
            if len(disc_idx) > len(best_disc_idx):
                best_disc_idx = disc_idx
                best_plane = np.array([a, b, c, d])
                best_plane_surface = plane_surface
                last_hit = i

    return best_plane_surface, best_plane, best_disc_idx, last_hit


def _to_camera_facing(plane):
    """Unit normal of ``plane`` flipped to face the camera (z < 0)."""
    normal = plane[:3] / np.linalg.norm(plane[:3])
    return -normal if normal[2] > 0 else normal


def estimate_suction_point(pc_img, mask=None, downsample_rate=5,
                           z_near=300.0, z_far=1050.0, vacuum_radius=30.0,
                           threshold=3.0, w_angle=0.25, w_dist=0.3, w_count=0.05, w_flat=0.4,
                           seal_min=0.98, seal_tol=5.0, holes=None, hole_margin=2.0,
                           rng=None, visualize=False, rgb_img=None, debug=None):
    """Estimate the optimal suction pose for a single target object.

    Evaluates geometrically-uniform candidate suction points over the object
    surface and picks the best by a weighted score (paper Eq. 2): surface angle
    (how top-facing the patch normal is) + proximity to the middle of the object
    + graspable-point count + flatness of the surroundings. Local flatness and a
    seal check (the whole cup disc must lie on the object, with valid depth, on
    the patch plane) gate candidate validity.

    Args:
        pc_img (np.ndarray): organized point cloud [H, W, 3] (mm) of the target
            object region (e.g. the object bbox crop).
        mask (np.ndarray, optional): boolean/0-1 object mask [H, W]; when given,
            only masked points are used (preferred over the depth-band filter).
        downsample_rate (int): uniform point subsampling factor (speed).
        z_near, z_far (float): metric depth band (mm) used when ``mask`` is None.
        vacuum_radius (float): suction cup radius (mm) for the disc search.
        threshold (float): plane inlier / flatness tolerance (mm).
        w_angle, w_dist, w_count, w_flat (float): score weights (sum to 1) for
            surface angle, proximity to the object centre, support count, and
            flatness of the patch's surroundings (see :func:`score_candidates`).
        seal_min (float): minimum fraction of the cup disc that must seal (see
            :func:`seal_ratio`); 0 disables the seal check.
        seal_tol (float): max distance (mm) of a sealing pixel from the cup plane.
        holes (np.ndarray, optional): [H, W] bool, known hole pixels (e.g. from the
            color image). They are left out of the surface (their depth is filled
            in, not measured), and the seal check rejects a cup within
            ``hole_margin`` mm of one. Only used with the seal check on.
        hole_margin (float): solid material (mm) required between the cup and a hole.
        rng (np.random.Generator, optional): only used by the RANSAC fallback.
        visualize (bool): draw the Open3D grasp visualization.
        rgb_img (np.ndarray, optional): color crop, only for visualization.
        debug (dict, optional): if given, filled with ``surface``, ``centroid`` (the
            surface point at the middle of the object),
            ``candidates`` (each scored), ``point``, ``normal`` for
            :func:`visualize_suction_steps`.

    Returns:
        SuctionPose: contact point (mm) and camera-facing unit normal.
        Returns ``None`` if the object has too few valid points, or (with the seal
        check on) no spot where the cup can seal.
    """
    in_band = (pc_img[:, :, 2] > z_near) & (pc_img[:, :, 2] < z_far)
    member = in_band & mask.astype(bool) if mask is not None else in_band
    if holes is not None:
        member = member & ~holes
    if mask is not None:
        surface = pc_img[member].reshape(-1, 3)
    else:
        surface = remove_background(pc_img, z_near, z_far)

    if len(surface) < 10:
        return None

    step = max(int(downsample_rate), 1)
    surface = surface[::step]

    # Evaluate geometrically-uniform candidates (plus one at the object centre) and
    # pick the best by the weighted score (paper Eq. 2): surface angle + proximity
    # to the middle of the object + graspable-point count + flatness.
    centroid = surface.mean(axis=0)
    model = _pixel_model(pc_img)
    region = mask.astype(bool) if mask is not None else in_band
    centre, scale, centre_seed = _object_centre(pc_img, region, member, model, surface)
    candidates = suction_candidates(surface, vacuum_radius=vacuum_radius, flat_tol=threshold,
                                    extra_seeds=centre_seed[None])
    if debug is not None:
        centroid = centre_seed          # plotted: the centre the score pulls toward
    if candidates:
        dists = centre_distance([c["point"] for c in candidates], centre, scale, model)
        for c, d in zip(candidates, dists):
            c["centre_dist"] = float(d)

    scored = score_candidates(candidates, centroid, w_angle, w_dist, w_count, w_flat, flat_tol=threshold)

    # Seal check, best score first: the first flat candidate whose cup disc seals
    # (no holes, edges or steps under it) wins, so usually only a few are checked.
    # Unchecked candidates keep seal=None.
    if seal_min <= 0:
        model = None
    for c in candidates:
        c["seal"] = None
    best = None
    if model is not None:
        for c in sorted(scored, key=lambda c: -c["score"]):
            c["seal"] = seal_ratio(pc_img, member, model, c["point"], c["normal"],
                                   vacuum_radius=vacuum_radius, tol=seal_tol,
                                   holes=holes, hole_margin=hole_margin)
            c["valid"] = c["seal"] >= seal_min
            if c["valid"]:
                best = c
                break
    elif scored:
        best = max(scored, key=lambda c: c["score"])

    if best is not None:
        point, normal = best["point"], best["normal"]
    elif model is not None:
        # nowhere on this object can the cup seal: better no grasp than a leaking one
        if debug is not None:
            debug.update(surface=surface, centroid=centroid, candidates=candidates,
                         disc=None, plane_surface=None, point=None, normal=None,
                         vacuum_radius=vacuum_radius)
        return None
    else:
        # fallback: densest flat cup-sized disc (RANSAC), else geometric center
        plane_surface, plane, disc_idx, iterations = ransac_disc_fit(
            surface, vacuum_radius=vacuum_radius, threshold=threshold, rng=rng)
        point = (np.mean(plane_surface[disc_idx], axis=0)
                 if iterations != 0 and plane_surface is not None else surface.mean(axis=0))
        normal = _to_camera_facing(plane)

    if debug is not None:
        # suction disc = surface points within the cup radius of the chosen point
        disc = surface[np.linalg.norm(surface - point, axis=1) < vacuum_radius]
        # largest coplanar flat region the chosen point lies on (its local plane)
        plane_surface = None
        if len(disc) >= 3:
            pl = fit_plane_lstsq(disc)
            plane_surface = surface[np.abs(surface @ pl[:3] + pl[3]) <= threshold]
        debug.update(surface=surface, centroid=centroid, candidates=candidates,
                     disc=disc, plane_surface=plane_surface,
                     point=point, normal=normal, vacuum_radius=vacuum_radius)

    if visualize:
        _visualize_suction(pc_img, surface, point, normal, rgb_img, vacuum_radius)

    return SuctionPose(point=point, normal=normal)


def visualize_suction_steps(debug, save_path=None, show=False, title=None, elev=62, azim=-90):
    """Plot how the suction point is chosen from the point cloud (paper Alg. 2).

    Near top-down 3-D scatter of: the object surface (grey), the largest flat
    plane the chosen point lies on (blue), the cup-sized suction disc at the
    chosen point (orange), the geometrically-uniform candidates colored by their
    weighted score (viridis; grey ✗ = too curved to seal), the object centre
    (black +), and the chosen contact point (red star, max score) with its
    camera-facing normal. Reads the ``debug`` dict populated by
    :func:`estimate_suction_point` (call it with ``debug={}``).

    Saves to ``save_path`` if given (headless-safe), and/or shows interactively.
    Returns the ``save_path``.
    """
    import matplotlib
    if save_path and not show:
        matplotlib.use("Agg")            # headless render-to-file
    import matplotlib.pyplot as plt

    surface = debug["surface"]
    point, normal = debug["point"], debug["normal"]
    centroid = debug.get("centroid")
    cands = debug.get("candidates") or []

    r = debug.get("vacuum_radius", 30.0)
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], s=4, c="lightgrey",
               label="object surface")
    if debug.get("plane_surface") is not None and len(debug["plane_surface"]):
        ps = debug["plane_surface"]
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], s=8, c="tab:blue", alpha=0.5,
                   label="largest flat plane")
    if debug.get("disc") is not None and len(debug["disc"]):
        d = debug["disc"]
        ax.scatter(d[:, 0], d[:, 1], d[:, 2], s=16, c="tab:orange",
                   label=f"suction disc (r={r:.0f}mm)")

    invalid = np.array([c["center"] for c in cands if not c["valid"]])
    valid = [c for c in cands if c["valid"]]
    if len(invalid):
        ax.scatter(invalid[:, 0], invalid[:, 1], invalid[:, 2], s=30, c="grey", marker="x",
                   alpha=0.4, label="candidate (not flat)")
    if valid:
        vpts = np.array([c["point"] for c in valid])
        vsc = np.array([c.get("score", 0.0) for c in valid])
        sc = ax.scatter(vpts[:, 0], vpts[:, 1], vpts[:, 2], s=60, c=vsc, cmap="viridis",
                        edgecolors="k", linewidths=0.4, label="suction candidates", depthshade=False)
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.12, label="suction score")
    if centroid is not None:
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], s=90, c="black", marker="+",
                   label="object centre")
    ax.scatter([point[0]], [point[1]], [point[2]], s=220, c="red", marker="*",
               label="chosen point (max score)", depthshade=False)
    ax.quiver(point[0], point[1], point[2], normal[0] * 40, normal[1] * 40, normal[2] * 40,
              color="red", linewidth=2)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.invert_zaxis()          # camera side (smaller z) up, so the camera-facing normal points up
    ax.view_init(elev=elev, azim=azim)   # near top-down (camera's-eye view of the bin)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or "Suction-point selection")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- viz
def rotation_between_vectors(vec1, vec2):
    """Rotation matrix aligning unit ``vec1`` to unit ``vec2`` (Rodrigues)."""
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if not np.any(v):
        return np.eye(3)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))


def _visualize_suction(pc_img, surface, point, normal, rgb_img, vacuum_radius):
    """Open3D rendering of the point cloud, suction point, and normal arrow."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_img.reshape(-1, 3))
    if rgb_img is not None:
        import cv2
        colors = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    sphere = o3d.geometry.TriangleMesh.create_sphere(4)
    sphere.translate(point, relative=False)
    sphere.paint_uniform_color([0, 1, 1])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30)
    frame.translate(point, relative=False)
    frame.rotate(rotation_between_vectors(np.array([0, 0, 1]), normal), center=point)

    disc = o3d.geometry.TriangleMesh.create_cylinder(radius=vacuum_radius, height=3)
    disc.translate(point, relative=False)
    disc.rotate(rotation_between_vectors(np.array([0, 0, 1]), normal), center=point)

    o3d.visualization.draw_geometries([pcd, sphere, frame, disc])
