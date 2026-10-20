"""End-to-end Seg2Grasp pipeline: segment -> select target -> grasp -> classify.

This is the single entry point that ties the three modules together (the paper's
modular design). Modules are injected so each can be used, swapped, or tested
independently:

    seg = Segmenter()                 # segmentation/segmenter.py
    clf = QwenClassifier()            # classification/qwen_classifier.py
    pipe = Seg2GraspPipeline(seg, clf)
    result = pipe.run(rgb, depth, pc)

`run` returns a :class:`Seg2GraspResult` with the chosen target's suction pose
(3-D point + normal), its category label, and everything needed to visualize or
command the robot. Segmentation and classification are heavy (GPU); the grasping
stage is analytic. Classification can be skipped for a grasp-only run.
"""
from dataclasses import dataclass

import numpy as np

from .grasping.hole_detection import detect_holes
from .grasping.suction_planner import _pixel_model, estimate_suction_point, SuctionPose
from .grasping.target_selection import rank_targets, TargetObject


@dataclass
class Seg2GraspResult:
    """Full pipeline output for one RGB-D frame."""
    target: TargetObject            # selected object (index, bbox, crops, centroid)
    suction: SuctionPose            # grasp point (mm) + camera-facing normal
    label: str = None               # predicted category (None if not classified)
    confidence: float = None        # classifier confidence
    masks: np.ndarray = None        # all segmentation masks [N, H, W]
    bboxes: np.ndarray = None       # all boxes [N, 4]
    seg_vis: np.ndarray = None      # segmentation visualization image (optional)
    holes: np.ndarray = None        # target's holes found in the color image [H, W] (optional)


class Seg2GraspPipeline:
    """Orchestrates the three Seg2Grasp modules over a single RGB-D frame.

    Args:
        segmenter: object with ``segment(rgb, depth) -> (masks, bboxes, vis)``.
        classifier: object with ``classify(scene_bgr, crop_bgr) -> dict`` (may be
            None to skip classification).
        vacuum_radius (float): suction-cup radius (mm) passed to the grasp planner.
        seal_min (float): minimum fraction of the cup disc that must seal
            (0 disables the planner's seal check).
        z_near, z_far (float): depth band (mm) of object points the grasp
            planner uses; points outside it are ignored.
        hole_detection (bool): find through-holes in the color image (see
            :mod:`.grasping.hole_detection`) so the cup is not placed over one.
        hole_margin (float): solid material (mm) required between the cup and a hole.
    """

    def __init__(self, segmenter, classifier=None, vacuum_radius=30.0, seal_min=0.98,
                 z_near=300.0, z_far=1050.0, hole_detection=True, hole_margin=2.0):
        self.segmenter = segmenter
        self.classifier = classifier
        self.vacuum_radius = vacuum_radius
        self.seal_min = seal_min
        self.z_near = z_near
        self.z_far = z_far
        self.hole_detection = hole_detection
        self.hole_margin = hole_margin

    def run(self, rgb, depth, pc, classify=True, visualize=False, rng=None):
        """Run the full pipeline on one frame.

        Args:
            rgb (np.ndarray): scene color image [H, W, 3] (BGR).
            depth (np.ndarray): raw metric depth [H, W] (mm) or normalized depth
                image, as expected by the segmenter.
            pc (np.ndarray): organized point cloud [H, W, 3] (mm).
            classify (bool): run the classifier on the chosen target.
            visualize (bool): show the Open3D grasp visualization.

        Returns:
            Seg2GraspResult for the most elevated object that has a suction
            point, or None if no segmented object has one.
        """
        masks, bboxes, seg_vis = self.segmenter.segment(rgb, depth)
        return self.grasp(rgb, pc, masks, bboxes, seg_vis=seg_vis, classify=classify,
                          visualize=visualize, rng=rng)

    def grasp(self, rgb, pc, masks, bboxes, seg_vis=None, classify=True, visualize=False, rng=None):
        """Target selection -> suction planning -> classification on given masks.

        The part of :meth:`run` after segmentation, so a caller can segment the
        next frame while this runs on the current one.
        """
        if masks is None or len(masks) == 0:
            return None

        # Most elevated object first; if the cup cannot seal anywhere on it (e.g. a
        # small or curved object), fall back to the next most elevated one. Objects
        # outside the depth band are skipped (e.g. the robot's own gripper in view of
        # a wrist camera): their masks can bleed onto the background behind them.
        # Holes are seen in the color image against the background, i.e. the pixels
        # outside every object's mask.
        model = None
        if self.hole_detection and self.seal_min > 0 and rgb is not None:
            model = _pixel_model(pc)
            background = ~np.any(np.asarray(masks, bool), axis=0)

        target = suction = holes = None
        for candidate in rank_targets(masks, bboxes, pc, rgb_img=rgb):
            if not self.z_near < candidate.centroid[2] < self.z_far:
                continue
            holes = holes_crop = None
            if model is not None:
                holes = detect_holes(rgb, masks[candidate.index], background,
                                     px_per_mm=model[0] / candidate.centroid[2])
                x0, y0, x1, y1 = candidate.bbox
                holes_crop = holes[y0:y1, x0:x1]
            suction = estimate_suction_point(
                candidate.pc_crop, mask=candidate.mask_crop, z_near=self.z_near, z_far=self.z_far,
                vacuum_radius=self.vacuum_radius, seal_min=self.seal_min, holes=holes_crop,
                hole_margin=self.hole_margin, rng=rng, visualize=visualize, rgb_img=candidate.rgb_crop)
            if suction is not None:
                target = candidate
                break
        if target is None:
            return None

        label, confidence = None, None
        if classify and self.classifier is not None and target.rgb_crop is not None:
            pred = self.classifier.classify(rgb, target.rgb_crop)
            label, confidence = pred.get("class_name"), pred.get("confidence")

        return Seg2GraspResult(
            target=target, suction=suction, label=label, confidence=confidence,
            masks=masks, bboxes=bboxes, seg_vis=seg_vis, holes=holes)
