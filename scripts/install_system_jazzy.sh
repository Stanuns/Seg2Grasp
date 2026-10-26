#!/usr/bin/env bash
# System-level dependencies for running Seg2Grasp under ROS 2 Jazzy on Ubuntu 24.04
# (tested on Jetson Thor / JetPack 7, L4T R39, CUDA 13.2). Needs root:
#
#   sudo bash scripts/install_system_jazzy.sh
#
# Installs:
#   - ROS 2 Jazzy desktop + ros-dev-tools (colcon, rosdep)
#   - CUDA toolkit (nvcc + headers) and python3.12-dev, for compiling detectron2 / MSDeformAttn
#   - RealSense D435i driver: ros-jazzy-realsense2-camera (apt, includes librealsense2) + udev rules
#   - Orbbec Gemini 305 driver build deps (OrbbecSDK_ROS2 in ~/ksdy_ws/orbbec_ws, via rosdep) + udev rules
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Please run with sudo"; exit 1; }
USER_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"
ORBBEC_SRC="${ORBBEC_SRC:-$USER_HOME/ksdy_ws/orbbec_ws/src/OrbbecSDK_ROS2}"
CUDA_PKG="${CUDA_PKG:-cuda-toolkit-13-2}"
ROS_MIRROR="${ROS_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu}"   # or http://packages.ros.org/ros2/ubuntu
export DEBIAN_FRONTEND=noninteractive

echo ">>> Ubuntu noble-updates source"
# ROS 2 Jazzy is built against noble-updates (e.g. boost *-dev); some Jetson images ship without it.
if ! grep -rqs "noble-updates" /etc/apt/sources.list /etc/apt/sources.list.d/; then
    cat > /etc/apt/sources.list.d/ubuntu-updates.sources <<'EOF'
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: noble-updates
Components: main universe restricted multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
fi

echo ">>> Locale + base tools"
apt-get update
apt-get install -y locales software-properties-common curl gnupg lsb-release \
    build-essential cmake git python3.12-dev python3-pip
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
add-apt-repository -y universe

echo ">>> ROS 2 apt source ($ROS_MIRROR)"
# Configured by hand instead of the ros2-apt-source .deb: github.com release downloads are
# unreachable from here, while raw.githubusercontent.com and the mirror are fast.
if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
    curl -sSL --fail https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
$ROS_MIRROR $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" > /etc/apt/sources.list.d/ros2.list
fi
apt-get update

echo ">>> ROS 2 Jazzy + RealSense driver + CUDA toolkit"
apt-get install -y ros-jazzy-desktop ros-dev-tools \
    ros-jazzy-realsense2-camera ros-jazzy-realsense2-description \
    "$CUDA_PKG"
# /usr/local/cuda -> the installed toolkit (the Jetson packages usually create it already)
[[ -e /usr/local/cuda ]] || ln -s "/usr/local/cuda-$(echo "${CUDA_PKG#cuda-toolkit-}" | tr - .)" /usr/local/cuda

echo ">>> RealSense udev rules"
curl -L --fail -o /etc/udev/rules.d/99-realsense-libusb.rules \
    https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules

echo ">>> Orbbec driver build deps + udev rules"
# The dependencies of orbbec_camera/package.xml, listed explicitly (rosdep needs extra network access)
apt-get install -y libgflags-dev nlohmann-json3-dev libgoogle-glog-dev libdw-dev libssl-dev \
    libopencv-dev libeigen3-dev libyaml-cpp-dev python3-psutil python3-tabulate python3-yaml \
    ros-jazzy-image-transport ros-jazzy-image-transport-plugins ros-jazzy-image-publisher \
    ros-jazzy-camera-info-manager ros-jazzy-cv-bridge ros-jazzy-diagnostic-updater \
    ros-jazzy-diagnostic-msgs ros-jazzy-statistics-msgs ros-jazzy-backward-ros \
    ros-jazzy-tf2-sensor-msgs ros-jazzy-message-filters ros-jazzy-xacro
if [[ -d "$ORBBEC_SRC" ]]; then
    cp "$ORBBEC_SRC/orbbec_camera/scripts/99-obsensor-libusb.rules" /etc/udev/rules.d/
else
    echo "!!! $ORBBEC_SRC not found: clone OrbbecSDK_ROS2 (v2-main) there and re-run"
fi
[[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]] || rosdep init || true
sudo -u "${SUDO_USER:-root}" rosdep update --rosdistro jazzy || echo "!!! rosdep update failed (optional)"

udevadm control --reload-rules && udevadm trigger
echo ">>> Done. Re-plug both cameras so the new udev rules apply."
