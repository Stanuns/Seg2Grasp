# Seg2Grasp 扩展说明（RealSense D435i + ROS 2）

以 ROS 2 节点运行：订阅 RealSense D435i 的彩色图与深度图 topic，实时输出目标物的吸取点（只用分割 + 抓取，不使用 Qwen 分类）。

环境：Ubuntu 22.04、ROS 2 Humble、`realsense2_camera`（realsense-ros 4.x）。

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
source install/setup.bash
source src/Seg2Grasp/venvs/seg/bin/activate
```

**方式一：一条命令同时启动相机和节点**

```bash
ros2 launch seg2grasp_ros seg2grasp.launch.py
ros2 launch seg2grasp_ros seg2grasp.launch.py roi:=20,140,440,500 vacuum_radius:=20.0
```

**方式二：相机和节点分开启动**

```bash
# 终端 1：相机
ros2 launch realsense2_camera rs_launch.py \
  pointcloud.enable:=true \
  depth_module.depth_profile:=848x480x30 \
  rgb_camera.color_profile:=848x480x30 \
  spatial_filter.enable:=true \
  temporal_filter.enable:=true \
  hole_filling_filter.enable:=true

# 终端 2：节点
ros2 run seg2grasp_ros suction_node
ros2 launch seg2grasp_ros seg2grasp.launch.py launch_camera:=false   # 或用 launch 只启动节点
```

| launch 参数 | 说明 |
|---|---|
| `launch_camera` | 是否同时启动相机，默认 `true` |
| `roi` | 彩色图裁剪区域 `y,x,h,w`，默认整幅图 |
| `vacuum_radius` | 吸盘半径（mm），默认 30；请设为实际吸盘半径（密封检查按此计算吸盘覆盖范围） |
| `seal_min` | 密封检查阈值：吸盘圆盘内能密封的像素比例下限，默认 0.98；设为 0 关闭检查 |
| `show` | 是否显示结果窗口，默认 `true` |

节点参数（`ros2 run ... --ros-args -p 参数:=值`）还包括 `color_topic`、`depth_topic`、`color_info_topic`、`depth_info_topic`，默认对应上面相机启动方式发布的 topic。

退出：在结果窗口按 `q` / Esc、关闭窗口，或在终端按 Ctrl+C。

## 处理方式

- 节点始终处理**最新**的一帧：处理期间到达的帧会被丢弃，处理完立即取下一帧最新数据。
- 每处理完一帧就刷新结果窗口，窗口始终显示最新结果（左下角显示单帧耗时，当前约 1.3 s/帧）。
- 深度图会用两个相机的内参和 TF 外参自动对齐到彩色图；若相机启用了 `align_depth.enable:=true`，将 `depth_topic` 设为 `/camera/camera/aligned_depth_to_color/image_raw` 即可跳过对齐。

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

## 避开弯折与褶皱

候选吸取点的评分中加入了"周边平整度"一项（权重 0.3）：计算吸盘半径 1.5 倍范围内的点到吸取面的偏离（90 分位），偏离越大得分越低。吸盘落在弯折、凸筋、褶皱上或紧挨着它们时，这一项会明显变差，从而优先选择平整区域。

## 密封检查

候选吸取点除了要求局部平整，还要求吸盘覆盖的整个圆盘都能密封：圆盘内的像素必须属于工件、有有效深度，且离吸取面的距离不超过 5 mm。覆盖率低于 `seal_min` 的候选会被排除；若目标物体上没有能密封的位置，输出 `no graspable object`。

- 能识别：吸盘越过工件边缘、跨折痕或台阶、深度相机能测到的开口。
- 不能识别：深度相机分辨不出的小孔。实测 D435i 在约 475 mm 距离上，对约 10 mm 的小孔给出的深度与周围表面连续，与是否开启滤波无关。

## 使用要求

- 工件表面到相机的距离需在 300–1050 mm 之间。
- 程序选择离相机最近（最高）的物体；画面边缘有杂物时，用 `roi` 只保留工件区域。
