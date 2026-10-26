#!/usr/bin/env bash
# Set up the SEGMENTATION + GRASPING environment for Seg2Grasp, using uv.
#
# Runs the Mask2Former (Swin-L) mask-proposal network and the analytic suction
# planner. It compiles two native extensions: detectron2's `_C` and the
# Mask2Former MSDeformAttn op. The checkpoint was trained on torch 1.9, but its
# weights load fine on torch 2.x.
#
# Target: Ubuntu 24.04 + ROS 2 Jazzy. The venv uses the SYSTEM Python 3.12, the
# interpreter ROS 2 Jazzy's rclpy is built for, so the ROS node can import both
# rclpy and torch. (On Humble/22.04 this was Python 3.10.)
#
# The CUDA build is picked from the machine, override with env vars:
#   aarch64 (Jetson Thor, JetPack 7 / CUDA 13.2): cu132 wheels, sm_110
#   x86_64  (e.g. Blackwell sm_120):              cu128 wheels, sm_120
#   TORCH_INDEX=cu130 TORCH_CUDA_ARCH_LIST=8.9 bash scripts/setup_seg_env.sh
# Requires `uv`, python3.12-dev, and a CUDA toolkit matching the torch build
# (CUDA_HOME, default /usr/local/cuda) for the native compiles.
#
# Usage:  bash scripts/setup_seg_env.sh
# Result: a venv at venvs/seg. Activate with `source venvs/seg/bin/activate`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/venvs/seg"
PYTHON="${PYTHON:-/usr/bin/python3.12}"
if [[ "$(uname -m)" == "aarch64" ]]; then
    TORCH_INDEX="${TORCH_INDEX:-cu132}"; CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-11.0}"   # Jetson Thor sm_110
    TORCH_PKGS="${TORCH_PKGS:-torch==2.12.1 torchvision==0.27.1}"
else
    TORCH_INDEX="${TORCH_INDEX:-cu128}"; CUDA_ARCH="${TORCH_CUDA_ARCH_LIST:-12.0}"   # Blackwell sm_120
    TORCH_PKGS="${TORCH_PKGS:-torch torchvision}"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc >/dev/null || { echo "!!! nvcc not found under $CUDA_HOME (install the CUDA toolkit)"; exit 1; }

if [[ ! -x "$VENV/bin/python" ]]; then
    echo ">>> Creating venv at $VENV with $PYTHON"
    uv venv --python "$PYTHON" "$VENV"
fi
# Keep colcon (ROS 2 workspace builds) from crawling the venv's site-packages.
touch "$REPO/venvs/COLCON_IGNORE"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">>> Installing PyTorch ($TORCH_INDEX)"
if [[ -n "${TORCH_WHL_MIRROR:-}" ]]; then
    # Slow download.pytorch.org: fetch the torch/torchvision wheels from a mirror (resumable) and take
    # their nvidia-* dependencies from the default index (UV_DEFAULT_INDEX, e.g. a PyPI mirror), e.g.
    #   TORCH_WHL_MIRROR=https://mirrors.nju.edu.cn/pytorch/whl \
    #   UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_seg_env.sh
    WHEELS="$REPO/venvs/wheels"; mkdir -p "$WHEELS"; files=()
    for spec in $TORCH_PKGS; do
        name="${spec%%==*}"; ver="${spec#*==}"
        f="$name-$ver+$TORCH_INDEX-cp312-cp312-manylinux_2_28_$(uname -m).whl"
        curl -L --fail --retry 5 -C - -o "$WHEELS/$f" "$TORCH_WHL_MIRROR/$TORCH_INDEX/${f//+/%2B}"
        files+=("$WHEELS/$f")
    done
    uv pip install "${files[@]}"
else
    # shellcheck disable=SC2086
    uv pip install $TORCH_PKGS --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi

echo ">>> Installing build tools + Seg2Grasp runtime deps (no transformers here)"
uv pip install setuptools wheel ninja
# numpy<2: the same major version as Ubuntu 24.04's python3-numpy used by ROS 2 Jazzy.
# typing_extensions / pyyaml / lark: imported by rclpy and the ROS message packages.
uv pip install "numpy<2" "opencv-python>=4.9" "open3d>=0.18" "pillow>=10.0" \
    "scipy>=1.11" "scikit-image>=0.22" "matplotlib>=3.7" tqdm pyyaml imageio \
    timm shapely pycocotools "fvcore>=0.1.5" "iopath>=0.1.9" cython \
    typing_extensions lark

echo ">>> Installing detectron2 from source (compiles _C for sm_${CUDA_ARCH})"
TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" FORCE_CUDA=1 \
    uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"

echo ">>> Compiling MSDeformAttn CUDA op (needs CUDA_HOME set)"
( cd "$REPO/third_party/mask2former/modeling/pixel_decoder/ops" && \
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" FORCE_CUDA=1 uv pip install --no-build-isolation . )

echo ">>> Installing seg2grasp (editable)"
uv pip install -e "$REPO" --no-deps

echo ">>> Done. Segmentation env ready at $VENV"
echo "    Note: the vendored legacy Mask2Former targets detectron2 0.6; if imports"
echo "    fail against detectron2 main, apply the small API fixes flagged at first run."
echo "    Next: place the checkpoint in data/checkpoints/segmentation/ (model_final.pth + config.yaml)"
