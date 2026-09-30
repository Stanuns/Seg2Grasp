"""Intel RealSense (D435i) capture for Seg2Grasp (RGB, metric depth, organized cloud).

Thin wrapper over ``pyrealsense2`` with the same interface as
:class:`~seg2grasp.camera.kinect.KinectCamera`. Depth is aligned to the color
stream, so every output shares the color camera's pixel grid and frame:
    rgb   [H, W, 3]  BGR uint8
    depth [H, W]     metric depth (mm)
    pc    [H, W, 3]  organized point cloud (mm), color-camera frame

A region-of-interest crop (the bin area) can be applied so downstream modules
only see the bin. ``pyrealsense2`` is imported lazily.
"""
import numpy as np


class RealSenseCamera:
    def __init__(self, roi=None, width=1280, height=720, fps=15, serial=None, warmup=30):
        """
        Args:
            roi (tuple, optional): (y_start, x_start, height, width) bin crop.
            width, height (int): stream resolution (color and depth).
            fps (int): stream frame rate.
            serial (str, optional): device serial, to pick one of several cameras.
            warmup (int): frames to drop at start so auto-exposure settles.
        """
        import pyrealsense2 as rs
        self._rs = rs
        self.roi = roi
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)

        self.depth_scale_mm = profile.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        self.align = rs.align(rs.stream.color)
        self.intrinsics = (profile.get_stream(rs.stream.color)
                           .as_video_stream_profile().get_intrinsics())

        # Pixel-ray grid of the color camera, for back-projecting aligned depth.
        # Undistorted pinhole model (D435 color streams report no distortion).
        k = self.intrinsics
        u, v = np.meshgrid(np.arange(k.width), np.arange(k.height))
        self._rays_x = ((u - k.ppx) / k.fx).astype(np.float32)
        self._rays_y = ((v - k.ppy) / k.fy).astype(np.float32)

        for _ in range(warmup):
            self.pipeline.wait_for_frames()

    def _crop(self, img):
        if self.roi is None:
            return img
        y, x, h, w = self.roi
        return img[y:y + h, x:x + w]

    def capture(self):
        """Grab one depth-to-color aligned frame.

        Returns:
            (rgb, depth, pc): BGR image, metric depth (mm), organized cloud (mm),
            all cropped to the ROI if one was set.
        """
        while True:
            frames = self.align.process(self.pipeline.wait_for_frames())
            color, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if color and depth_frame:
                break
        rgb = np.asanyarray(color.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale_mm
        pc = np.stack([self._rays_x * depth, self._rays_y * depth, depth], axis=-1)
        return self._crop(rgb), self._crop(depth), self._crop(pc)

    def pixel_of(self, point):
        """Project a 3-D point (mm, color-camera frame) to full-frame pixel (u, v)."""
        k = self.intrinsics
        return (int(round(point[0] / point[2] * k.fx + k.ppx)),
                int(round(point[1] / point[2] * k.fy + k.ppy)))

    def close(self):
        self.pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
