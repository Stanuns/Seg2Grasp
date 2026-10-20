# Seg2Grasp 扩展说明（RealSense D435i / Orbbec Gemini 305 + ROS 2）

以 ROS 2 节点运行：订阅相机（RealSense D435i 或 Orbbec Gemini 305）的彩色图与深度图 topic，实时输出目标物的吸取点（只用分割 + 抓取，不使用 Qwen 分类）。

环境：Ubuntu 22.04、ROS 2 Humble、`realsense2_camera`（realsense-ros 4.x）、`orbbec_camera`（OrbbecSDK_ROS2，使用 Gemini 305 时需要）。

## 环境准备

```bash
cd ~/ksdy_ws/ws_seg2grasp/src/Seg2Grasp
bash scripts/setup_seg_env.sh        # 创建 venvs/seg（无需运行 setup_qwen_env.sh）
bash scripts/download_weights.sh     # 下载分割模型权重到 data/checkpoints/segmentation/

cd ~/ksdy_ws/ws_seg2grasp
source /opt/ros/humble/setup.bash
colcon build --packages-select seg2grasp_ros
```

> 不要用 `--symlink-install`：该模式生成的节点入口固定为 `/usr/bin/python3`，找不到 venv 中的 torch。
> 只修改 `seg2grasp/` 下的算法代码无需重新编译；修改 `ros/seg2grasp_ros/` 下的节点或 launch 文件后需重新 `colcon build`。

## 运行

每个新终端先加载环境（顺序不能乱，venv 必须最后激活）：

```bash
cd ~/ksdy_ws/ws_seg2grasp
source /opt/ros/humble/setup.bash
source ~/ksdy_ws/embodied_ws/install/setup.bash   # realsense2_camera 所在工作区
source ~/ksdy_ws/orbbec_ws/install/setup.bash     # orbbec_camera 所在工作区
source install/setup.bash
source src/Seg2Grasp/venvs/seg/bin/activate
```

**方式一：一条命令同时启动相机和节点**

用 `camera` 参数选择相机：`d435i`（默认）或 `gemini305`。

```bash
ros2 launch seg2grasp_ros seg2grasp.launch.py                       # RealSense D435i
ros2 launch seg2grasp_ros seg2grasp.launch.py camera:=gemini305     # Orbbec Gemini 305
ros2 launch seg2grasp_ros seg2grasp.launch.py roi:=20,140,440,500 vacuum_radius:=20.0
```

两台相机不要同时启动：两者默认都发布 `camera_link`、`camera_color_optical_frame` 等同名 TF 坐标系。

**方式二：相机和节点分开启动**

```bash
# 终端 1：相机（D435i）
ros2 launch realsense2_camera rs_launch.py \
  pointcloud.enable:=true \
  depth_module.depth_profile:=848x480x30 \
  rgb_camera.color_profile:=848x480x30 \
  spatial_filter.enable:=true \
  temporal_filter.enable:=true \
  hole_filling_filter.enable:=true

# 终端 1：相机（Gemini 305）
ros2 launch orbbec_camera gemini_301_series.launch.py \
  enable_colored_point_cloud:=true \
  depth_registration:=true

# 终端 2：节点（只启动节点，camera 与终端 1 的相机一致）
ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false
ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false camera:=gemini305
ros2 run seg2grasp_ros suction_node   # 或直接运行节点（默认订阅 D435i 的 topic）
```

Gemini 305 启动时 `enable_colored_point_cloud:=true` 会让驱动自动打开 `depth_registration`（这里也显式写出）：`/camera/depth/image_raw` 已对齐到彩色图（坐标系 `camera_color_optical_frame`，分辨率与彩色图相同），节点直接使用，不再做深度配准。

| launch 参数 | 说明 |
|---|---|
| `camera` | 使用的相机：`d435i`（默认）或 `gemini305`；决定启动的相机驱动、订阅的 topic 和 `z_near` / `z_far` 默认值 |
| `launch_camera` | 是否同时启动相机，默认 `true` |
| `roi` | 彩色图裁剪区域 `y,x,h,w`，默认整幅图 |
| `vacuum_radius` | 吸盘半径（mm），默认 30；请设为实际吸盘半径（密封检查按此计算吸盘覆盖范围） |
| `seal_min` | 密封检查阈值：吸盘圆盘内能密封的像素比例下限，默认 0.98；设为 0 关闭检查 |
| `show` | 是否显示结果窗口，默认 `true` |
| `fp16` | 分割模型用 float16 推理，默认 `true`（比 float32 快约 3 倍，结果几乎相同） |
| `seg_min_size` | 分割输入图像短边像素，默认 `0`（即模型默认 800）；设为 600 / 480 更快，但分割更粗 |
| `z_near` / `z_far` | 抓取使用的物体深度范围（mm），范围外的点被忽略；留空则按相机取默认值：`d435i` 为 300 / 2000，`gemini305` 为 70 / 500 |

节点参数（`ros2 run ... --ros-args -p 参数:=值`）还包括 `color_topic`、`depth_topic`、`color_info_topic`、`depth_info_topic`。launch 会按 `camera` 自动设置；直接 `ros2 run` 时默认是 D435i 的 topic，使用 Gemini 305 时需手动指定：

| 节点参数 | D435i（默认） | Gemini 305 |
|---|---|---|
| `color_topic` | `/camera/camera/color/image_raw` | `/camera/color/image_raw` |
| `depth_topic` | `/camera/camera/depth/image_rect_raw` | `/camera/depth/image_raw` |
| `color_info_topic` | `/camera/camera/color/camera_info` | `/camera/color/camera_info` |
| `depth_info_topic` | `/camera/camera/depth/camera_info` | `/camera/depth/camera_info` |
| `z_near` | `300.0` | `70.0` |
| `z_far` | `2000.0` | `500.0` |

退出：在结果窗口按 `q` / Esc、关闭窗口，或在终端按 Ctrl+C。

## 处理方式

- 节点始终处理**最新**的一帧：处理期间到达的帧会被丢弃，处理完立即取下一帧最新数据。
- 两段流水线：后台线程分割新一帧的同时，主线程规划上一帧的吸取点。默认设置下 `/seg2grasp/suction_pose` 约 2 Hz、单帧延迟约 0.6 s（RTX 5080 Laptop 实测）；`seg_min_size:=600` 约 2.5 Hz，`480` 约 2.8 Hz。
- 每处理完一帧就刷新结果窗口，窗口始终显示最新结果（左下角显示输出频率和延迟）。
- 深度图会用两个相机的内参和 TF 外参自动对齐到彩色图；若深度已与彩色图对齐（坐标系和分辨率都相同），则跳过对齐。D435i 若启用了 `align_depth.enable:=true`，将 `depth_topic` 设为 `/camera/camera/aligned_depth_to_color/image_raw` 即可；Gemini 305 按上面的方式启动时深度已对齐。

## 输出

| Topic | 类型 | 说明 |
|---|---|---|
| `/seg2grasp/suction_pose` | `geometry_msgs/PoseStamped` | 吸取点位姿，坐标系 `camera_color_optical_frame`，单位 m；姿态的 z 轴为吸取接近方向（指向物体表面内） |
| `/seg2grasp/result_image` | `sensor_msgs/Image`（bgr8） | 标注后的结果图 |

终端日志每帧输出一行 JSON：

```json
{"point_mm": [-89.8, -30.7, 515.5], "normal": [0.1446, 0.0287, -0.9891], "pixel": [320, 212], "target": 0}
```

- `point_mm`：吸取点，单位 mm，彩色相机光学坐标系（x 向右、y 向下、z 向前）。
- `normal`：吸取面法向（朝向相机，`z < 0`）。
- `pixel`：吸取点在原图中的像素位置。
- `target`：选中物体的编号；无可抓取物体时输出 `no graspable object`。

## 吸取点评分

候选吸取点按 2/3 吸盘半径的间距均匀分布在工件表面，另在工件中心处额外加一个候选点。每个候选点按四项加权打分：

| 项 | 权重 | 含义 |
|---|---|---|
| 居中 | 0.30 | 离工件中心越近越好 |
| 周边平整度 | 0.40 | 吸盘周边越平整越好，避开弯折、凸筋、褶皱 |
| 朝向 | 0.25 | 吸取面越正对相机越好 |
| 支撑点数 | 0.05 | 吸盘范围内的有效点越多越好 |

得分从高到低依次做密封检查，第一个通过的候选点即为输出的吸取点。

## 居中选点

工件中心取分割掩码的像素重心，掩码内没有有效深度的像素（反光、孔洞处）也计入，因此深度缺失不会把中心拉偏。候选点到中心的距离以工件等效半径（√(面积/π)）为单位：在中心为 0，到边缘约为 1，居中得分为 `1 − 距离`。

在平整的工件上，其余三项各候选点得分相近，吸取点会落在工件中间。只有中间不能吸（有孔、弯折、凸筋、倾斜，或密封检查不通过）时，才会退到离中心尽量近、且能密封的位置。

## 避开弯折与褶皱

"周边平整度"计算吸盘半径 1.5 倍范围内的点到吸取面的偏离（90 分位），偏离越大得分越低。吸盘落在弯折、凸筋、褶皱上或紧挨着它们时，这一项会明显变差，从而优先选择平整区域。

偏离量会先减去该工件自身的噪声水平（所有候选点偏离量的 10 分位），再与 3 mm 比较。因此平面上的深度噪声不影响选点，只有真正的弯折、凸筋会扣分。

## 密封检查

候选吸取点除了要求局部平整，还要求吸盘覆盖的整个圆盘都能密封：圆盘内的像素必须属于工件、有有效深度，且离吸取面的距离不超过 5 mm。覆盖率低于 `seal_min` 的候选会被排除；若目标物体上没有能密封的位置，输出 `no graspable object`。

- 能识别：吸盘越过工件边缘、跨折痕或台阶、深度相机能测到的开口。
- 不能识别：深度相机分辨不出的小孔。实测 D435i 在约 475 mm 距离上，对约 10 mm 的小孔给出的深度与周围表面连续，与是否开启滤波无关。

## 使用要求

- 工件表面到相机的距离需在 `z_near`–`z_far` 之间（D435i 默认 300–2000 mm，Gemini 305 默认 70–500 mm）。
- 程序选择离相机最近（最高）的物体；画面边缘有杂物时，用 `roi` 只保留工件区域。
