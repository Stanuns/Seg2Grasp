#!/usr/bin/env bash
# Set up the SEGMENTATION + GRASPING environment for Seg2Grasp on NVIDIA Jetson
# (JetPack 6 / CUDA 12.6, e.g. Jetson Orin NX), using the system Python 3.10.
#
# Same result as scripts/setup_seg_env.sh, adapted for Jetson:
#   - venv from the system python3 (3.10), so the ROS 2 Humble rclpy loads in it;
#   - PyTorch from NVIDIA's JetPack 6 wheels (jetson-ai-lab index), not cu128;
#   - detectron2 `_C` and the Mask2Former MSDeformAttn op compiled for sm_87
#     (Orin), with MAX_JOBS limited so the build fits in the board's memory;
#   - opencv-python < 4.12 (newer releases require numpy 2; we keep numpy < 2).
# No Qwen environment: the ROS node only segments + grasps.
#
# Usage:  bash scripts/setup_seg_env_jetson.sh
# Result: a venv at venvs/seg. Activate with `source venvs/seg/bin/activate`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/venvs/seg"
CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-8.7}"    # Orin (sm_87)
TORCH_INDEX="${TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"

echo ">>> Creating Python 3.10 venv at $VENV"
python3 -m venv "$VENV"
# Keep colcon (ROS 2 workspace builds) from crawling the venv's site-packages.
touch "$REPO/venvs/COLCON_IGNORE"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip setuptools wheel

echo ">>> Installing PyTorch (JetPack 6 wheels from $TORCH_INDEX)"
pip install torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX"

echo ">>> Installing build tools + Seg2Grasp runtime deps (no transformers here)"
pip install "numpy<2" "opencv-python>=4.9,<4.12" "open3d>=0.18" "pillow>=10.0" \
    "scipy>=1.11" "scikit-image>=0.22" "matplotlib>=3.7" tqdm pyyaml imageio \
    timm shapely pycocotools "fvcore>=0.1.5" "iopath>=0.1.9" cython ninja

echo ">>> Installing detectron2 from source (compiles _C for sm_${CUDA_ARCH/./})"
TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" FORCE_CUDA=1 \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

echo ">>> Compiling MSDeformAttn CUDA op"
( cd "$REPO/third_party/mask2former/modeling/pixel_decoder/ops" && \
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" FORCE_CUDA=1 python setup.py build install )

echo ">>> Installing seg2grasp (editable)"
pip install -e "$REPO" --no-deps

echo ">>> Checking the install"
( cd /tmp && python -c "
import torch, detectron2, MultiScaleDeformableAttention, seg2grasp
from detectron2 import _C
assert torch.cuda.is_available(), 'CUDA not available: torch is not the JetPack build'
print('torch', torch.__version__, '| CUDA arch', torch.cuda.get_arch_list(), '| detectron2', detectron2.__version__)
" )

echo ">>> Done. Segmentation env ready at $VENV"
echo "    Next: place the checkpoint in data/checkpoints/segmentation/ (model_final.pth + config.yaml)"
