# Seg2Grasp 扩展说明（RealSense D435i / Orbbec Gemini 305 + ROS 2）

以 ROS 2 节点运行：订阅相机（RealSense D435i 或 Orbbec Gemini 305）的彩色图与深度图 topic，实时输出目标物的吸取点（只用分割 + 抓取，不使用 Qwen 分类）。

环境：Ubuntu 22.04、ROS 2 Humble、`realsense2_camera`（realsense-ros 4.x）、`orbbec_camera`（OrbbecSDK_ROS2，使用 Gemini 305 时需要）。

## 环境准备

```bash
cd ~/ksdy_ws/ws_seg2grasp/src/Seg2Grasp
bash scripts/setup_seg_env.sh        # 创建 venvs/seg（无需运行 setup_qwen_env.sh）；Jetson 上改用 setup_seg_env_jetson.sh，见下文
bash scripts/download_weights.sh     # 下载分割模型权重到 data/checkpoints/segmentation/

cd ~/ksdy_ws/ws_seg2grasp
source /opt/ros/humble/setup.bash
colcon build --packages-select seg2grasp_ros
```

> 不要用 `--symlink-install`：该模式生成的节点入口固定为 `/usr/bin/python3`，找不到 venv 中的 torch。
> 只修改 `seg2grasp/` 下的算法代码无需重新编译；修改 `ros/seg2grasp_ros/` 下的节点或 launch 文件后需重新 `colcon build`。

### Jetson Orin NX（JetPack 6）

`scripts/setup_seg_env.sh` 面向 x86 + Blackwell 显卡（cu128），在 Jetson 上不能直接使用，改用 `scripts/setup_seg_env_jetson.sh`。已在 Jetson Orin NX Super（JetPack 6.x / L4T R36.5、CUDA 12.6、Ubuntu 22.04、ROS 2 Humble）上验证：

```bash
cd ~/ksdy_ws/ws_seg2grasp/src/Seg2Grasp
bash scripts/setup_seg_env_jetson.sh   # 创建 venvs/seg，最后检查 CUDA 可用、架构为 sm_87
bash scripts/download_weights.sh       # 或手动把 model_final.pth + config.yaml 放到 data/checkpoints/segmentation/
```

之后与上面相同，`colcon build --packages-select seg2grasp_ros`。脚本可用环境变量覆盖默认值：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | `8.7` | 编译架构（Orin 为 8.7） |
| `TORCH_INDEX` | `https://pypi.jetson-ai-lab.io/jp6/cu126` | Jetson PyTorch wheel 源（JetPack 6 / CUDA 12.6） |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA 工具链 |
| `MAX_JOBS` | `4` | 并行编译数；内存不足导致编译被杀时调小 |

与 `setup_seg_env.sh` 的区别：

| 项 | `setup_seg_env.sh` | `setup_seg_env_jetson.sh` |
|---|---|---|
| 创建 venv | `uv venv --python 3.10` | `python3 -m venv`（系统 Python 3.10，与 ROS 2 Humble 的 rclpy 兼容） |
| PyTorch | cu128 官方 wheel | NVIDIA JetPack 6 专用 wheel：torch 2.8.0 / torchvision 0.23.0（jetson-ai-lab 源，CUDA 12.6） |
| 编译架构 | `TORCH_CUDA_ARCH_LIST=12.0` | `8.7`（Orin），并设 `MAX_JOBS=4`，防止编译时内存不足 |
| opencv | 不限版本 | `opencv-python<4.12`（新版要求 numpy 2，与 `numpy<2` 冲突） |
| 分割权重 | `download_weights.sh` | 相同；也可手动把 `model_final.pth` + `config.yaml` 放到 `data/checkpoints/segmentation/` |
| Qwen 环境 | `setup_qwen_env.sh` | 不安装（ROS 节点不使用分类） |

> `~/.local` 中若有 pip 装的非 Jetson 版 torch（如 `+cu130`），在 JetPack 6 上 `torch.cuda.is_available()` 为 `False`；venv 不受影响，但不要用系统 `python3` 运行本项目。

Jetson 上的使用差异：

- 加载环境时不需要 `source ~/ksdy_ws/embodied_ws/install/setup.bash`：`realsense2_camera` 由 apt（`ros-humble-realsense2-camera`）安装在 `/opt/ros/humble` 中。
- 建议加 `seg_min_size:=480`。实测（40W 模式）：

  | 设置 | `suction_pose` 频率 | 延迟 |
  |---|---|---|
  | D435i 单开，默认 800 | 约 0.3 Hz | 约 3.2 s |
  | D435i 单开，`seg_min_size:=480` | 约 0.6 Hz | 约 2.2 s |
  | `camera:=both`，`seg_min_size:=480` | 各约 0.9 Hz | 约 1.3–1.4 s |

  单开两行为早期实测（`seg_input:=rgbd`）。`camera:=both` 一行为当前默认设置的实测：`seg_input:=rgb`、相机 15 fps、D435i 驱动端对齐深度（此前 `rgbd`、30 fps、节点配准时为 0.43 / 0.46 Hz、约 2.6 s）。两个节点共用 GPU，分割是瓶颈；`seg_input:=rgbd` 时每帧 GPU 耗时约翻倍。

- `camera:=both` 时两个模型共占约 13 GB 内存（16 GB 板），运行时请关闭其他程序。

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

用 `camera` 参数选择相机：`d435i`（默认）、`gemini305`，或 `both` 同时使用两台。

```bash
ros2 launch seg2grasp_ros seg2grasp.launch.py                       # RealSense D435i
ros2 launch seg2grasp_ros seg2grasp.launch.py camera:=gemini305     # Orbbec Gemini 305
ros2 launch seg2grasp_ros seg2grasp.launch.py camera:=both          # 两台相机同时使用
ros2 launch seg2grasp_ros seg2grasp.launch.py roi:=20,140,440,500 vacuum_radius:=20.0
```

**两台相机同时使用**：每台相机始终以自己的名字运行（不论单开还是双开），topic、TF 坐标系和吸取点结果互不冲突；每台相机一个节点 `/seg2grasp/<相机名>`，各自加载分割模型、各开一个结果窗口（标题含节点名）。

| | D435i | Gemini 305 |
|---|---|---|
| 相机 topic | `/d435i/color/image_raw`、`/d435i/aligned_depth_to_color/image_raw` 等 | `/gemini305/color/image_raw`、`/gemini305/depth/image_raw` 等 |
| TF 坐标系 | `d435i_link`、`d435i_color_optical_frame`、`d435i_depth_optical_frame` 等 | `gemini305_link`、`gemini305_color_optical_frame` 等 |
| 节点 | `/seg2grasp/d435i` | `/seg2grasp/gemini305` |
| 吸取点 | `/seg2grasp/d435i/suction_pose`（坐标系 `d435i_color_optical_frame`） | `/seg2grasp/gemini305/suction_pose`（坐标系 `gemini305_color_optical_frame`） |
| 结果图 | `/seg2grasp/d435i/result_image` | `/seg2grasp/gemini305/result_image` |

- `camera:=both` 时 `roi`、`vacuum_radius`、`z_near` 等参数对两个节点同时生效（`z_near` / `z_far` 留空的按各自相机取默认值）。两台相机需要不同的 `roi` 等参数时，分两个终端各启动一次：`camera:=d435i` 和 `camera:=gemini305`。
- 两个节点共用一块 GPU，实测各约 1.5 Hz（单开约 2.2 Hz）。

**方式二：相机和节点分开启动**

```bash
# 终端 1：相机（D435i）
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=/ camera_name:=d435i \
  pointcloud.enable:=true \
  depth_module.depth_profile:=848x480x15 \
  rgb_camera.color_profile:=848x480x15 \
  spatial_filter.enable:=true \
  temporal_filter.enable:=true \
  hole_filling_filter.enable:=true \
  align_depth.enable:=true

# 终端 1：相机（Gemini 305）
ros2 launch orbbec_camera gemini_301_series.launch.py \
  camera_name:=gemini305 \
  enable_colored_point_cloud:=true \
  depth_registration:=true \
  color_width:=848 color_height:=530 color_fps:=15 color_format:=YUYV \
  depth_width:=848 depth_height:=530 depth_fps:=15

# 终端 2：节点（只启动节点，camera 与终端 1 的相机一致）
ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false
ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false camera:=gemini305
ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false camera:=both seg_min_size:=480
ros2 run seg2grasp_ros suction_node   # 或直接运行节点：/seg2grasp/d435i，订阅 D435i 的 topic
```

相机必须带上 `camera_name`（D435i 还需 `camera_namespace:=/`），topic 和坐标系才是上表中的名字。

两台相机都以 15 fps 运行：节点每秒只处理约 1 帧，多余的帧仍要在节点里反序列化、做彩色/深度同步，占用 CPU。Gemini 305 需显式写 `color_format:=YUYV`，否则驱动在 15 fps 下会选 MJPG（多一道解码）。

D435i 需 `align_depth.enable:=true`：节点订阅 `/d435i/aligned_depth_to_color/image_raw`（驱动已对齐到彩色图），不再自己做深度配准（Jetson 上每帧约 95 ms）。若相机未开启对齐，节点会一直等待该 topic；此时可改回 `-p depth_topic:=/d435i/depth/image_rect_raw -p depth_info_topic:=/d435i/depth/camera_info`，由节点配准。

Gemini 305 启动时 `enable_colored_point_cloud:=true` 会让驱动自动打开 `depth_registration`（这里也显式写出）：`/gemini305/depth/image_raw` 已对齐到彩色图（坐标系 `gemini305_color_optical_frame`，分辨率与彩色图相同），节点直接使用，不再做深度配准。

| launch 参数 | 说明 |
|---|---|
| `camera` | 使用的相机：`d435i`（默认）、`gemini305` 或 `both`；决定启动的相机驱动、节点、订阅的 topic 和 `z_near` / `z_far` 默认值 |
| `launch_camera` | 是否同时启动相机，默认 `true` |
| `roi` | 彩色图裁剪区域 `y,x,h,w`，默认整幅图；`camera:=both` 时对两台相机都生效 |
| `vacuum_radius` | 吸盘半径（mm），默认 15（机械臂吸盘直径约 30 mm）。吸盘属于机械臂，两台相机用同一个值。请设为实际吸盘半径：设大了会找不到足够大的平整区域（大量 `no graspable object`，如 30 mm 时两台相机在冲压支架上几乎都找不到吸取点），平整度评估的范围也会跨过弯折 |
| `seal_min` | 密封检查阈值：吸盘圆盘内能密封的像素比例下限，默认 0.98；设为 0 关闭检查 |
| `show` | 是否显示结果窗口，默认 `true` |
| `fp16` | 分割模型用 float16 推理，默认 `true`（比 float32 快约 3 倍，结果几乎相同） |
| `seg_input` | 分割网络的输入：`rgb`（默认，只用彩色图）或 `rgbd`（彩色图和深度图各跑一遍网络再融合）。`rgb` 的 GPU 耗时约为 `rgbd` 的一半；在示例场景上两者的掩码平均 IoU 约 0.96，`rgb` 偶尔多分出几个物体。若出现漏分、粘连，改回 `rgbd` |
| `seg_min_size` | 分割输入图像短边像素，默认 `0`（即模型默认 800）；设为 600 / 480 更快，但分割更粗 |
| `z_near` / `z_far` | 抓取使用的物体深度范围（mm），范围外的点被忽略；留空则按相机取默认值：`d435i` 为 300 / 2000，`gemini305` 为 70 / 500 |
| `hole_detection` | 用彩色图识别工件上的孔洞，吸盘不覆盖孔洞，默认 `true`（见下文"孔洞识别"） |
| `hole_margin` | 吸盘边缘与孔洞之间至少保留的实体宽度（mm），默认 2 |
| `target_area_weight` | 选择目标物时面积的权重，0–1，默认 0.5（见下文"目标物选择"）。0 = 只选离相机最近（最高）的物体；1 = 只选分割面积最大的物体 |
| `w_angle` / `w_dist` / `w_count` / `w_flat` | 吸取点评分的四项权重（见下文"吸取点评分"），默认 0.25 / 0.3 / 0.05 / 0.4。只有比例起作用，建议保持和为 1。例如更靠中心：`w_dist:=0.5 w_flat:=0.25 w_angle:=0.2`；`camera:=both` 时对两个节点同时生效 |

节点参数（`ros2 run ... --ros-args -p 参数:=值`）还包括 `color_topic`、`depth_topic`、`color_info_topic`、`depth_info_topic`。launch 会按 `camera` 自动设置；直接 `ros2 run` 时节点为 `/seg2grasp/d435i`、使用 D435i 的 topic，使用 Gemini 305 时需改节点名并指定 topic（`--ros-args -r __node:=gemini305 -p color_topic:=... ...`）：

| 节点参数 | D435i（默认） | Gemini 305 |
|---|---|---|
| `color_topic` | `/d435i/color/image_raw` | `/gemini305/color/image_raw` |
| `depth_topic` | `/d435i/aligned_depth_to_color/image_raw` | `/gemini305/depth/image_raw` |
| `color_info_topic` | `/d435i/color/camera_info` | `/gemini305/color/camera_info` |
| `depth_info_topic` | `/d435i/aligned_depth_to_color/camera_info` | `/gemini305/depth/camera_info` |
| `z_near` | `300.0` | `70.0` |
| `z_far` | `2000.0` | `500.0` |

退出：在结果窗口按 `q` / Esc、关闭窗口，或在终端按 Ctrl+C。

## 处理方式

- 节点始终处理**最新**的一帧：处理期间到达的帧会被丢弃，处理完立即取下一帧最新数据。
- 两段流水线：后台线程分割新一帧的同时，主线程规划上一帧的吸取点。默认设置下单台相机的 `suction_pose` 约 2 Hz、单帧延迟约 0.6 s（RTX 5080 Laptop 实测）；`seg_min_size:=600` 约 2.5 Hz，`480` 约 2.8 Hz。
- 每处理完一帧就刷新结果窗口，窗口始终显示最新结果（左下角显示输出频率和延迟）。
- 若深度已与彩色图对齐（坐标系和分辨率都相同），节点直接使用；按上面的方式启动时两台相机都是这种情况（D435i `align_depth.enable:=true`，Gemini 305 `depth_registration:=true`）。否则用两个相机的内参和 TF 外参自动对齐到彩色图。
- `seg_input:=rgbd` 时，分割网络的深度输入在 1/4 分辨率上修补空洞（Jetson 上 92 → 10 ms，掩码与全分辨率修补几乎相同）；只影响网络输入，规划用的点云不受影响。
- 分割后的掩码清理（去重合并、外接框）只在掩码外接框的交集内计算重叠，结果与逐像素全图计算完全相同，掩码多时（如 Gemini 305 近距离 20 多个）每帧从约 400 ms 降到约 40 ms。

## 输出

每台相机一组（`<相机名>` 为 `d435i` 或 `gemini305`）：

| Topic | 类型 | 说明 |
|---|---|---|
| `/seg2grasp/<相机名>/suction_pose` | `geometry_msgs/PoseStamped` | 吸取点位姿，坐标系为该相机的 `<相机名>_color_optical_frame`，单位 m；姿态的 z 轴为吸取接近方向（指向物体表面内） |
| `/seg2grasp/<相机名>/result_image` | `sensor_msgs/Image`（bgr8） | 标注后的结果图 |

终端日志每帧输出一行 JSON，行首的 `[seg2grasp.<相机名>]` 表示来自哪台相机：

```json
{"point_mm": [-89.8, -30.7, 515.5], "normal": [0.1446, 0.0287, -0.9891], "pixel": [320, 212], "target": 0}
```

- `point_mm`：吸取点，单位 mm，彩色相机光学坐标系（x 向右、y 向下、z 向前）。
- `normal`：吸取面法向（朝向相机，`z < 0`）。
- `pixel`：吸取点在原图中的像素位置。
- `target`：选中物体的编号；无可抓取物体时输出 `no graspable object`。

## 目标物选择

先选目标物，再在它上面找吸取点。深度范围 `z_near`–`z_far` 内的物体按得分从高到低依次尝试，某个物体上找不到能密封的位置时换下一个：

得分 = (1 − `target_area_weight`) × 高度 + `target_area_weight` × 面积

- 高度 = 最近物体的深度 / 该物体的深度：最近（最高）的物体为 1，越远越低。
- 面积 = 该物体的分割像素数 / 最大物体的分割像素数。

只按高度选时（`target_area_weight:=0`），画面边缘比工件略近的小物体（杂物、机械臂自身的部件）会被选中，吸取点落到角落。默认 0.5 时面积起主要作用，这类小物体不再被选中。

实测（D435i，斜视桌面上的几根冲压支架，约 455 mm 处画面左下角有一小块反光物体）：`target_area_weight:=0` 时 3 帧中有 1 帧选中左下角的小物体，0.3 / 0.5 / 1 时都选中工件。注意分割偶尔把相邻的两根工件分成一个掩码，面积加权时这种掩码面积最大、会被优先选中；吸取点仍须通过密封检查，只会落在其中一根工件的平整处。

## 吸取点评分

候选吸取点按 2/3 吸盘半径的间距均匀分布在工件表面，另在工件中心处额外加一个候选点。每个候选点按四项加权打分：

| 项 | 参数 | 默认权重 | 含义 |
|---|---|---|---|
| 居中 | `w_dist` | 0.30 | 离工件中心越近越好 |
| 周边平整度 | `w_flat` | 0.40 | 吸盘周边越平整越好，避开弯折、凸筋、褶皱 |
| 朝向 | `w_angle` | 0.25 | 吸取面越正对相机越好 |
| 支撑点数 | `w_count` | 0.05 | 吸盘范围内的有效点越多越好 |

权重可用同名 launch 参数调整（如 `w_dist:=0.5 w_flat:=0.25 w_angle:=0.2`）。调高 `w_dist` 时吸取点更靠中心，但更可能落在中心附近的弯折、凸筋旁；密封检查不受权重影响。

得分从高到低依次做密封检查，第一个通过的候选点即为输出的吸取点。

## 居中选点

工件中心取分割掩码的像素重心，掩码内没有有效深度的像素（反光、孔洞处）也计入，因此深度缺失不会把中心拉偏。候选点到中心的距离以工件等效半径（√(面积/π)）为单位：在中心为 0，到边缘约为 1，居中得分为 `1 − 距离`。

在平整的工件上，其余三项各候选点得分相近，吸取点会落在工件中间。只有中间不能吸（有孔、弯折、凸筋、倾斜，或密封检查不通过）时，才会退到离中心尽量近、且能密封的位置。

## 避开弯折与褶皱

"周边平整度"计算吸盘半径 1.5 倍范围内的点到吸取面的偏离（90 分位），偏离越大得分越低。吸盘落在弯折、凸筋、褶皱上或紧挨着它们时，这一项会明显变差，从而优先选择平整区域。

偏离量会先减去该工件自身的噪声水平（所有候选点偏离量的 10 分位），再与 3 mm 比较。因此平面上的深度噪声不影响选点，只有真正的弯折、凸筋会扣分。

## 密封检查

候选吸取点除了要求局部平整，还要求吸盘覆盖的整个圆盘都能密封：圆盘内的像素必须属于工件、有有效深度，且离吸取面的距离不超过 5 mm。覆盖率低于 `seal_min` 的候选会被排除；若目标物体上没有能密封的位置，输出 `no graspable object`。

深度异常值：金属反光处偶尔会出现一片远离工件的错误深度（实测 D435i 在约 520 mm 的支架上出现过约 1.4 m 的一片），这片点自身平整，会通过平整度和密封检查。因此离工件深度中位数超过工件自身横向尺寸的点不参与选点。

- 能识别：吸盘越过工件边缘、跨折痕或台阶、深度相机能测到的开口。
- 深度相机分辨不出的小孔：实测 D435i 在约 475 mm 距离上，对约 10 mm 的小孔给出的深度与周围表面连续，与是否开启滤波无关；Gemini 305 在约 280 mm 距离上对约 8 mm 的圆孔也是如此。这类孔洞由下面的"孔洞识别"从彩色图中找出。

## 孔洞识别

分割掩码通常把工件上的孔洞也包含在内，深度图又测不出小孔，因此仅靠深度无法发现吸盘盖住了孔。彩色图可以，用两种线索，结果取并集：

1. 颜色：透过孔洞看到的是工件后面的背景（如桌面）。
2. 暗度：斜着看或孔后光线不足时，看到的是孔内壁的阴影，明显比周围工件暗。亮度低于工件亮度中位数一半的像素判为孔洞（工件本身太暗时不用这条线索）。

颜色线索：

- 背景颜色取自目标周围、不属于任何分割物体的像素；工件颜色取自目标掩码内的像素（用中位数，不受孔洞和反光影响）。
- 颜色用色度（R、G 各除以 R+G+B）比较，与亮度无关：透过孔洞看到的背景被工件遮挡、偏暗，色度仍沿"工件色 → 背景色"方向，甚至更远（如木纹桌面由黄变棕）。
- 掩码内的像素在"工件色（0）→ 背景色（1）"轴上超过 0.8 即判为孔洞；去除直径小于 3 mm 的噪点。只看朝背景色方向的偏移：金属反射出的其他颜色（如偏蓝）不会被当成孔洞。掩码溢出到背景上的部分也会被标出，等同于"不属于工件"。
- 密封检查中，吸盘半径加 `hole_margin` 范围内只要有孔洞像素，该候选点即判为不能密封（孔洞再小也会漏气）；孔洞像素也不参与平整度拟合（其深度是周围表面填补的，不是实测值）。
- 结果窗口中目标物体上识别出的孔洞以红色轮廓显示。
- 限制：
  - 某条线索判出的孔洞超过工件像素的 25% 时（工件与背景颜色相近，或工件光照不均），忽略该线索；两条都不可用时退回仅用深度的密封检查。
  - 很小的孔（图像中不足约 5 像素，Gemini 305 在 350 mm 处约 4 mm）可能识别不到。
  - 工件阴影一侧的边缘有时也会被暗度线索标出，这些位置本来就是工件边缘，不影响选点。
  - 只在密封检查开启（`seal_min > 0`）时生效。

实测（Gemini 305，带矩形窗口和多个圆孔的冲压支架）：约 280 mm、20 帧，关闭孔洞识别时有 13 帧的吸盘盖住了圆孔，开启后 0 帧，20 帧都找到吸取点；约 350 mm、工件反光偏蓝的另一视角下，矩形窗口和两个圆孔同样被识别，金属表面无误判。D435i 斜视约 520 mm 时圆孔几乎全黑，仅靠颜色线索识别不到，约一半帧的吸盘压到圆孔；加入暗度线索后圆孔被识别，30 帧中吸盘不再压到圆孔。每帧增加约 20–30 ms 的规划时间（与分割并行，不降低输出频率）。

## 使用要求

- 工件表面到相机的距离需在 `z_near`–`z_far` 之间（D435i 默认 300–2000 mm，Gemini 305 默认 70–500 mm）。
- 程序按高度和分割面积选择目标物（见"目标物选择"）；画面边缘有杂物时，也可用 `roi` 只保留工件区域。
- 中心深度不在 `z_near`–`z_far` 内的物体不作为目标，例如腕部相机视野中的夹爪本体。机器人自身的吸盘若在该范围内，因其表面弯曲通常不会被选中；如需彻底排除，可调大 `z_near` 或用 `roi` 裁掉。
