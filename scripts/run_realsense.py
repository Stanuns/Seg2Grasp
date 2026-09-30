#!/usr/bin/env python3
"""Live Seg2Grasp suction-point detection with an Intel RealSense D435i.

Segmentation + grasping only (no classifier). Each frame prints the chosen
target's suction point (mm, color-camera frame), its camera-facing normal and
the pixel it lands on.

    source venvs/seg/bin/activate
    python scripts/run_realsense.py                      # window + terminal control (see below)
    python scripts/run_realsense.py --once --no-gui      # one frame, print + save, exit

Controls: next frame = Enter in the terminal or any key in the window; quit =
'q'+Enter in the terminal, q/Esc in the window, closing the window, or Ctrl+C.

Point/normal are in the RealSense color optical frame (x right, y down, z forward).
"""
import argparse
import json
import os
import queue
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seg2grasp import paths
from seg2grasp.pipeline import Seg2GraspPipeline
from seg2grasp.segmentation.segmenter import Segmenter
from seg2grasp.segmentation.visualize import annotate_result

WINDOW = "Seg2Grasp RealSense (q=quit)"


def _read_terminal(commands):
    """Forward terminal lines to the main loop ('' = next frame, 'q' = quit)."""
    for line in sys.stdin:
        commands.put(line.strip().lower())
    commands.put("q")  # stdin closed


def wait_next(commands):
    """Wait for the user to request the next frame; returns False to quit.

    Polls instead of ``cv2.waitKey(0)`` so the terminal, the window's close
    button and Ctrl+C all stay responsive.
    """
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key != 0xFF:
            return True
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
        try:
            cmd = commands.get_nowait()
        except queue.Empty:
            continue
        return cmd not in ("q", "quit", "exit")


def main():
    ap = argparse.ArgumentParser(description="Live Seg2Grasp loop (RealSense D435i, seg + grasp).")
    ap.add_argument("--roi", default=None, help="bin crop 'y,x,h,w' (default: full frame)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--serial", default=None, help="RealSense serial (if several are connected)")
    ap.add_argument("--vacuum-radius", type=float, default=30.0, help="suction cup radius (mm)")
    ap.add_argument("--once", action="store_true", help="process a single frame and exit")
    ap.add_argument("--no-gui", action="store_true", help="no window; save the annotated image instead")
    args = ap.parse_args()

    from seg2grasp.camera.realsense import RealSenseCamera
    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    camera = RealSenseCamera(roi=roi, width=args.width, height=args.height,
                             fps=args.fps, serial=args.serial)
    pipe = Seg2GraspPipeline(Segmenter(), classifier=None, vacuum_radius=args.vacuum_radius)
    out_dir = os.path.join(paths.OUTPUT_ROOT, "realsense")
    os.makedirs(out_dir, exist_ok=True)

    commands = queue.Queue()
    if not args.no_gui:
        threading.Thread(target=_read_terminal, args=(commands,), daemon=True).start()
        print(">>> Live loop. Enter = next frame, q+Enter = quit (or q/Esc in the window).")
    else:
        print(">>> Running. Ctrl+C to stop.")
    i = 0
    try:
        while True:
            rgb, depth, pc = camera.capture()
            result = pipe.run(rgb, depth, pc, classify=False)
            if result is None:
                print("no graspable object")
                vis = rgb
            else:
                p, n = result.suction.point.astype(float), result.suction.normal.astype(float)
                u, v = camera.pixel_of(p)
                print(json.dumps({"point_mm": p.round(1).tolist(), "normal": n.round(4).tolist(),
                                  "pixel": [u, v], "target": result.target.index}))
                vis = annotate_result(rgb, result, pc)

            if args.no_gui:
                path = os.path.join(out_dir, f"frame_{i:04d}.png")
                cv2.imwrite(path, vis)
                print(f"    saved {path}")
            else:
                cv2.imshow(WINDOW, vis)
                if not wait_next(commands):
                    break
            i += 1
            if args.once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
