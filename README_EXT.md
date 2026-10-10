# Seg2Grasp 扩展说明（RealSense D435i）

输入 Intel RealSense D435i 的 RGB-D 数据，输出目标物的吸取点（只用分割 + 抓取，不使用 Qwen 分类）。

## 环境准备

```bash
bash scripts/setup_seg_env.sh        # 创建 venvs/seg（无需运行 setup_qwen_env.sh）
source venvs/seg/bin/activate
uv pip install pyrealsense2
bash scripts/download_weights.sh     # 下载分割模型权重到 data/checkpoints/segmentation/
```

## 运行

```bash
source venvs/seg/bin/activate
python scripts/run_realsense.py                       # 弹窗显示，逐帧处理
python scripts/run_realsense.py --once --no-gui       # 只处理一帧，打印结果并存图后退出
python scripts/run_realsense.py --roi 20,140,440,500  # 只处理指定区域 y,x,h,w
```

| 参数 | 说明 |
|---|---|
| `--roi y,x,h,w` | 裁剪区域，默认整幅图 |
| `--width` / `--height` | 相机分辨率，默认 848×480 |
| `--fps` | 帧率，默认 15 |
| `--serial` | 多台 RealSense 时指定序列号 |
| `--vacuum-radius` | 吸盘半径（mm），默认 30 |
| `--once` | 只处理一帧 |
| `--no-gui` | 不弹窗，标注图存到 `data/outputs/realsense/` |

| 操作 | 终端 | 图像窗口 |
|---|---|---|
| 下一帧 | 回车 | 任意键 |
| 退出 | `q` + 回车，或 Ctrl+C | `q` / Esc，或关闭窗口 |

## 输出

每帧输出一行 JSON：

```json
{"point_mm": [-57.9, 31.0, 373.0], "normal": [-0.0633, -0.0134, -0.9979], "pixel": [331, 298], "target": 0}
```

- `point_mm`：吸取点，单位 mm，D435i 彩色相机光学坐标系（x 向右、y 向下、z 向前）。用于机械臂时需再乘以手眼标定矩阵。
- `normal`：吸取面法向（朝向相机，`z < 0`）。
- `pixel`：吸取点在原图中的像素位置。
- `target`：选中物体的编号；无可抓取物体时输出 `no graspable object`。

## 使用要求

- 工件表面到相机的距离需在 300–1050 mm 之间。
- 程序选择离相机最近（最高）的物体；画面边缘有杂物时，用 `--roi` 只保留工件区域。
