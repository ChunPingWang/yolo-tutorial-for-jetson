# ml-vision

Jetson Orin Nano 8GB 上的即時影像識別專案 (YOLO + TensorRT)。

## 硬體 / 軟體

| 項目 | 規格 |
|------|------|
| 裝置 | NVIDIA Jetson Orin Nano 8GB (Tegra Orin) |
| JetPack | 6.x (L4T R36.5) |
| CUDA / TensorRT | 12.6 / 10.3 (apt 已裝) |
| Python | 系統 `/usr/bin/python3.10` (不要用 linuxbrew 的 3.14) |
| 框架 | PyTorch 2.5 (Jetson wheel) + Ultralytics YOLOv11 |

## 一鍵安裝

```bash
./setup.sh
```

本機 `~/.local` 已有 CUDA-ready 的 torch 2.8 + torchvision 0.23 + TensorRT 10.3，所以 setup.sh 只做：
1. 確認 `/usr/bin/python3.10`
2. `pip install --user` 補裝 ultralytics / opencv-python / onnx
3. 跑驗證（印出 CUDA 是否就緒）

> ⚠️ **不要 `pip install torch`** — PyPI 的 aarch64 wheel 是 CPU-only，會覆蓋掉現有 CUDA 版本。
> Jetson CUDA torch 是 NVIDIA 特製的，要從 `developer.download.nvidia.com/compute/redist/jp/v61/pytorch/` 下載。

## 使用

### 1. 圖檔推論 (不需要相機，現在就能跑)

```bash
# 直接抓 ultralytics 內建測試圖
python src/detect_image.py --source https://ultralytics.com/images/bus.jpg --save-json

# 跑自己的圖
python src/detect_image.py --source data/my_photo.jpg --conf 0.4
python src/detect_image.py --source data/ --imgsz 640        # 整個資料夾
```

輸出在 `output/detect_image/`。

### 2. 即時相機

```bash
python src/detect_camera.py                          # 自動偵測 USB / CSI
python src/detect_camera.py --source 0               # /dev/video0 (USB webcam)
python src/detect_camera.py --source csi --csi-id 0  # CSI MIPI (IMX219 / IMX477)
python src/detect_camera.py --headless --save output/run.mp4 --max-frames 300
```

按 `q` 退出。沒接顯示器加 `--headless`。

### 3. 匯出 TensorRT engine (加速 2-3x)

```bash
# FP16 engine (Jetson Orin 原生支援，預設)
python src/export_tensorrt.py --weights models/yolo11n.pt

# INT8 量化 (需要校準資料)
python src/export_tensorrt.py --weights models/yolo11n.pt --int8 --data coco128.yaml
```

匯出後在偵測腳本指定 `--weights models/yolo11n.engine` 就會用 TensorRT 推論。

### 4. Benchmark

```bash
python src/benchmark.py                                  # 預設 .pt
python src/benchmark.py --weights models/yolo11n.engine  # TRT engine 對比
```

印出 p50 / p95 / p99 延遲與 FPS。

## 專案結構

```
ml-vision/
├── README.md
├── requirements.txt
├── setup.sh                  # 一鍵環境
├── src/
│   ├── detect_image.py       # 圖檔 / 資料夾推論
│   ├── detect_camera.py      # 即時相機 (USB + CSI)
│   ├── export_tensorrt.py    # .pt → .engine
│   └── benchmark.py          # 延遲 / FPS 量測
├── models/                   # 權重 (.pt / .engine)
├── data/                     # 輸入影像
└── output/                   # 標註結果
```

## 效能 (本機實測, Jetson Orin Nano 8GB, 640x640, YOLOv11n)

| 格式 | 平均延遲 | p99 | FPS | 加速比 | 模型大小 |
|------|---------|-----|-----|--------|---------|
| PyTorch .pt (FP32) | 35.8 ms | 37.6 ms | **27.9** | 1.00× | 5.4 MB |
| TensorRT .engine (FP16) | 20.8 ms | 21.2 ms | **48.1** | 1.72× | 8.3 MB |
| TensorRT .engine (INT8) | 18.4 ms | 19.6 ms | **54.5** | 1.95× | 5.4 MB |

INT8 校準：coco128 (128 張 COCO 樣本，ultralytics 會自動下載)。

**精度觀察** (bus.jpg, 5 個 GT 物件)
- FP32：5 偵測 ✓
- FP16：4 偵測（漏掉信心 0.62 的邊緣 person）
- INT8：4 偵測（同 FP16，沒額外掉）

INT8 對 YOLOv11n 這種小模型加速有限（~13% over FP16），因為瓶頸在 NMS / 後處理而非 matmul。換大模型 (YOLOv11s/m) 增益會更明顯。

為了穩定的數字，跑前先進 MAXN 模式：

```bash
sudo nvpmodel -m 0   # MAXN
sudo jetson_clocks   # 鎖最高頻
```

## 常用診斷

```bash
tegrastats           # GPU/CPU/記憶體即時狀態 (Jetson 專用，取代 nvidia-smi 的 mem 顯示)
ls /dev/video*       # 確認 USB 相機掛載
gst-launch-1.0 nvarguscamerasrc ! fakesink   # 測 CSI MIPI 相機
```
