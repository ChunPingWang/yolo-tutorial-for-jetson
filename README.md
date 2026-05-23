# 社區住戶安全系統-使用 YOLO 影像辨識技術

從零到能用 GPU 跑 49 FPS 的 YOLO 物件偵測。

本專案是一個**邊緣 AI 概念驗證 (PoC)**：在 NVIDIA Jetson Orin Nano 8GB 上跑 YOLOv11，並用 TensorRT 把 PyTorch 模型加速到接近兩倍 FPS。本文除了使用說明，也整理了從硬體、深度學習到推論加速的基礎知識，讓你看完就懂為什麼每一步要這樣做。

---

## 目錄

1. [基礎知識](#基礎知識)
   - [什麼是 Jetson？](#什麼是-jetson)
   - [影像識別三大任務](#影像識別三大任務)
   - [YOLO 是什麼](#yolo-是什麼)
   - [完整推論流程](#完整推論流程)
   - [為什麼需要 TensorRT](#為什麼需要-tensorrt)
   - [精度與量化（FP32/FP16/INT8）](#精度與量化fp32fp16int8)
2. [硬體 / 軟體規格](#硬體--軟體規格)
3. [環境安裝](#環境安裝)
4. [動手做](#動手做)
   - [Step 1：第一張圖](#step-1第一張圖)
   - [Step 2：看懂結果](#step-2看懂結果)
   - [Step 3：接相機即時推論](#step-3接相機即時推論)
   - [Step 4：用 TensorRT 加速](#step-4用-tensorrt-加速)
   - [Step 5：INT8 量化再壓榨](#step-5int8-量化再壓榨)
   - [Step 6：怎麼讀 benchmark](#step-6怎麼讀-benchmark)
5. [效能與精度結果](#效能與精度結果)
6. [疑難排解 / 踩雷紀錄](#疑難排解--踩雷紀錄)
7. [進階主題](#進階主題)
8. [延伸閱讀](#延伸閱讀)

---

## 基礎知識

### 什麼是 Jetson？

Jetson 是 NVIDIA 推出的**邊緣運算模組**系列，把 GPU、CPU、影像處理器整合在一張郵票大小的 SoM (System on Module)上，專門設計給機器人、智慧相機、無人機這類「不能 / 不適合連雲」的場景。

本機是 **Jetson Orin Nano 8GB**：
- **CPU**：6 核 Arm Cortex-A78AE @ 1.5 GHz
- **GPU**：1024 顆 CUDA Core + 32 顆 Tensor Core (Ampere 架構)，整合在 SoC 內，**和 CPU 共用同一塊 LPDDR5 記憶體 (8 GB)**
- **AI 算力**：稀疏推論可達 67 TOPS（INT8）
- **功耗**：7-15W 可調

跟桌機顯卡的關鍵差別：
| | 桌機 GPU (RTX) | Jetson |
|---|---|---|
| GPU 記憶體 | 獨立 VRAM | **與 CPU 共用 RAM** |
| `nvidia-smi` | 完整支援 | 只顯示驅動，記憶體要用 `tegrastats` |
| 開發板擴充 | PCIe | **CSI MIPI 相機 / GPIO / I²C** |
| 部署目標 | 雲 / 工作站 | **產品內嵌** |

「共用記憶體」是雙面刃：好處是 CPU↔GPU 零拷貝、省 PCIe 傳輸；壞處是 RAM 緊張時 GPU 跟 CPU 互相搶。

### 影像識別三大任務

「影像識別」是個籠統詞，實務上分三種，難度遞增：

| 任務 | 輸出 | 例子 | 常見模型 |
|------|------|------|---------|
| **分類** (Classification) | 整張圖一個標籤 | 「這是一隻貓」 | ResNet, EfficientNet |
| **物件偵測** (Detection) | 多個 bbox + 標籤 | 「左上有一輛車，右下有一個人」 | **YOLO**, Faster R-CNN |
| **分割** (Segmentation) | 每個像素一個標籤 | 「這些像素屬於車、這些屬於路面」 | Mask R-CNN, SAM |

本 PoC 做的是**物件偵測**：給一張圖，輸出 N 個矩形框（bounding box）和每個框的類別 + 信心分數。

### YOLO 是什麼

YOLO = **You Only Look Once**，2015 年由 Joseph Redmon 提出。在 YOLO 之前，主流物件偵測（如 Faster R-CNN）是「兩階段」：先找出可能有物件的候選區域，再分類每個區域。慢且複雜。

YOLO 的關鍵洞見：把整張圖丟進 CNN **一次**就直接輸出所有 bbox 和類別 — **single-shot detector**。速度提升一個數量級，足以做即時推論。

歷代演進（重點）：
- **YOLOv1–v3** (2015-2018, Redmon)：奠基者，DarkNet 框架
- **YOLOv5** (2020, Ultralytics)：搬到 PyTorch，工程友好
- **YOLOv8** (2023, Ultralytics)：anchor-free，一個架構吃分類/偵測/分割/姿態
- **YOLOv11** (2024, Ultralytics)：本專案用的版本，C3k2 / SPPF / C2PSA 模組，比 v8 同尺寸更準也更快

每個版本都有不同尺寸：`n`(nano) / `s`(small) / `m`(medium) / `l`(large) / `x`(extra-large)。本專案用 `yolo11n.pt`（5.4 MB），參數量最少、速度最快，適合 Orin Nano 這種邊緣裝置。

YOLO 在 **COCO 資料集**上預訓練，能偵測 80 種類別（人、車、動物、家具⋯）。類別清單在 [`coco.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)。

### 完整推論流程

從相機到「畫上 bbox 的圖」，中間發生了什麼事：

```
┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────┐   ┌──────────┐   ┌────────┐
│ 相機 RAW │ → │ 影像解碼  │ → │ 預處理      │ → │ 模型推論 │ → │ 後處理    │ → │ 視覺化  │
│ (BGR)   │   │ (OpenCV) │   │ resize/norm│   │ (GPU)   │   │ NMS 解碼  │   │ 畫 bbox │
└─────────┘   └──────────┘   └────────────┘   └─────────┘   └──────────┘   └────────┘
```

每一步在做什麼：

1. **預處理 (Preprocess)**
   - 把任意尺寸的 BGR 影像 letterbox 縮放到模型輸入大小（本專案 640×640）
   - BGR → RGB
   - uint8 [0,255] → float32 [0,1]
   - HWC → CHW → batch (NCHW)
2. **模型推論 (Inference)**
   - 一次前向傳播，輸出 `[batch, anchors, 4+1+80]`
   - 4 = bbox 座標、1 = objectness、80 = 類別機率
3. **後處理 (Post-process)**
   - 用信心閾值 `conf` 砍掉低分 anchor
   - **NMS (Non-Maximum Suppression)**：同個物件常被多個 anchor 框到，用 IoU 閾值合併重疊框
   - 把座標從 640×640 縮回原圖尺寸
4. **視覺化**：在原圖畫 bbox + 類別 + 信心

ultralytics 的 `model.predict()` 把 1-4 包成一個 call。我們在 `src/detect_image.py` 看到的就是這層 API。

### 為什麼需要 TensorRT

PyTorch 在訓練時很好用，但拿 `.pt` 直接推論不是最快的。原因：
1. **動態圖**：PyTorch 每次 forward 都重新跟蹤計算圖，有額外開銷
2. **沒做硬體特化**：通用 CUDA kernel，沒針對特定 GPU 型號優化
3. **多餘的精度**：訓練要 FP32 才不會發散，但推論用 FP16 / INT8 也夠

**TensorRT** 是 NVIDIA 出的推論專用編譯器，做了：
- **圖優化**：合併 Conv+BN+ReLU 成單一 kernel，減少 GPU 啟動次數
- **Kernel auto-tuning**：對你的特定 GPU 型號試不同 kernel 實作，選最快的
- **精度降階**：自動把適合的層轉成 FP16 / INT8
- **記憶體重用**：靜態分析所有 tensor 生命週期，重用 buffer

代價：產出的 `.engine` 跟 **硬體型號 + TensorRT 版本**綁定，換機器要重新匯出（auto-tuning 結果不能搬）。所以在 Orin Nano 上匯出的 engine 不能直接搬到 Orin AGX 跑，反之亦然。

從 PyTorch 到 TensorRT 的完整路徑：
```
.pt (PyTorch)  →  .onnx (ONNX 中介格式)  →  .engine (TensorRT)
```
ultralytics 的 `model.export(format="engine")` 一行命令搞定，內部會跑 PyTorch → ONNX → TensorRT 兩段轉換。

### 精度與量化（FP32/FP16/INT8）

神經網路每個權重和啟動值原本是 **32 位元浮點數 (FP32)**。**量化 (Quantization)** 是把這些數字壓成更窄的型別：

| 精度 | 位元 | 範圍 | 主要場景 |
|------|------|------|---------|
| FP32 | 32 | ±3.4×10³⁸ | 訓練、最高精度推論 |
| FP16 | 16 | ±6.5×10⁴ | 推論加速首選，精度幾乎不掉 |
| INT8 | 8 | ±127 | 極致加速，需要校準 |
| INT4 | 4 | ±7 | 大模型 LLM，CV 較少用 |

為什麼量化能加速？
1. **記憶體頻寬**：權重變 1/4 大小，從 DRAM 載入的時間也是 1/4
2. **Tensor Core**：Jetson Orin 的 Tensor Core 對 FP16 / INT8 有專用指令，吞吐量 FP32 的 2×–4×
3. **快取友好**：更小的 tensor 更容易塞進 L2 cache

但 INT8 不是免費午餐：
- **動態範圍小**：FP32 能表示 10³⁸，INT8 只到 127。要做「**校準 (calibration)**」決定每層的縮放係數
- **校準集**：餵 100-500 張代表性的影像進去跑一次，TRT 觀察每層啟動值分佈，算出最佳 scale
- **精度損失**：通常 mAP 掉 0.5-2%，邊緣情況可能漏偵測（我們實測就是這樣）

本專案的 INT8 校準用 **coco128**（128 張 COCO 樣本），ultralytics 會自動下載。

---

## 硬體 / 軟體規格

| 項目 | 規格 |
|------|------|
| 裝置 | NVIDIA Jetson Orin Nano 8GB (Tegra Orin) |
| JetPack | 6.x (L4T R36.5) |
| CUDA / TensorRT | 12.6 / 10.3 (apt 已裝) |
| Python | 系統 `/usr/bin/python3.10` (不要用 linuxbrew 的 3.14) |
| 框架 | PyTorch 2.8 (CUDA aarch64) + Ultralytics YOLOv11 |

---

## 環境安裝

```bash
./setup.sh
```

本機 `~/.local` 已有 CUDA-ready 的 torch 2.8 + torchvision 0.23 + TensorRT 10.3，所以 setup.sh 只做：
1. 確認 `/usr/bin/python3.10`
2. `pip install --user` 補裝 ultralytics / opencv-python / onnx
3. 跑驗證（印出 CUDA 是否就緒）

> ⚠️ **不要 `pip install torch`** — PyPI 的 aarch64 wheel 是 CPU-only，會覆蓋掉現有 CUDA 版本。Jetson CUDA torch 是 NVIDIA 特製的，要從 `developer.download.nvidia.com/compute/redist/jp/v61/pytorch/` 下載。

驗證通過會看到：
```
torch       : 2.8.0  cuda=True  device=Orin
opencv      : 4.11.0
ultralytics : 8.4.53
tensorrt    : 10.3.0
```

---

## 動手做

### Step 1：第一張圖

不需要相機，直接跑 ultralytics 官方測試圖：
```bash
python src/detect_image.py --source https://ultralytics.com/images/bus.jpg --save-json
```

第一次跑會自動下載 5.4 MB 的 `yolo11n.pt` 權重到 `models/`。輸出在 `output/detect_image/bus.jpg`。

正常看到：
```
[✓] 推論 1 張圖, 共 5 個物件, 耗時 0.84s (1.2 img/s)
```

> 「1.2 img/s」是含模型載入+CUDA warmup 的時間。實際每幀推論在 36ms（27 FPS），看 [Step 6](#step-6怎麼讀-benchmark)。

### Step 2：看懂結果

打開 `output/detect_image/detections.json`，會看到：
```json
[
  {
    "file": "bus.jpg",
    "size": [810, 1080],
    "detections": [
      {
        "class": "bus",
        "confidence": 0.940,
        "bbox_xyxy": [3.83, 229.36, 796.21, 728.40]
      },
      ...
    ]
  }
]
```

- `size`：原圖 [寬, 高]
- `class`：COCO 80 類之一
- `confidence`：0~1，越高越確定
- `bbox_xyxy`：左上角 (x1, y1) 到右下角 (x2, y2)，**原圖座標系**

其他常見 bbox 格式：
- `xywh`：[中心 x, 中心 y, 寬, 高]
- `xyxyn`：xyxy 但歸一化到 [0,1]，與輸入尺寸無關
- YOLO 訓練標註：`xywhn`（中心 + 寬高，全部歸一化）

### Step 3：接相機即時推論

接 USB webcam 後：
```bash
python src/detect_camera.py                          # 自動偵測
python src/detect_camera.py --source 0               # 指定 /dev/video0
python src/detect_camera.py --headless --save output/run.mp4 --max-frames 300
```

如果是 CSI MIPI 相機（IMX219 / IMX477）：
```bash
python src/detect_camera.py --source csi --csi-id 0
```

CSI 走 GStreamer pipeline 而不是 V4L2，因為要經過 `nvarguscamerasrc`（NVIDIA Argus 影像處理）才能拿到原始幀。pipeline 在 `src/detect_camera.py:gstreamer_csi_pipeline` 可以看到。

按 `q` 退出。畫面左上會顯示 FPS。

### Step 4：用 TensorRT 加速

```bash
python src/export_tensorrt.py --weights models/yolo11n.pt
```

第一次匯出會花 **5-10 分鐘**（TRT 在做 kernel auto-tuning），之後產出 `models/yolo11n.engine`。換 engine 推論：
```bash
python src/detect_image.py --weights models/yolo11n.engine --source data/my.jpg
python src/detect_camera.py --weights models/yolo11n.engine
```

預期速度約 PyTorch 的 1.7-2 倍。

### Step 5：INT8 量化再壓榨

```bash
python src/export_tensorrt.py --weights models/yolo11n.pt --int8 --data coco128.yaml
```

匯出時 ultralytics 會：
1. 下載 coco128 校準集（128 張 COCO 樣本，約 6 MB）
2. 跑校準（每層觀察啟動值分佈，算 scale）
3. 用 INT8 編譯 engine

完成後跟 FP16 比，本專案實測再快約 13%（看下面表）。

### Step 6：怎麼讀 benchmark

```bash
python src/benchmark.py --weights models/yolo11n.pt --warmup 20 --iters 200
```

輸出：
```
推論延遲 (ms)   平均  35.83   p50  35.78   p95  36.47   p99  37.56
輸送量            27.9 FPS
```

**怎麼解讀**：
- **平均 / p50（中位數）**：典型情況的延遲
- **p95 / p99**：尾部延遲，反映卡頓嚴重程度
  - p99 接近平均 → 穩定（如本例 37.6 vs 35.8）
  - p99 比平均高很多 → 有 GC、頁面換出、熱降頻等問題
- **FPS = 1000 / 平均延遲**（單緒）

`--warmup 20` 很關鍵：前幾次推論會跑 cuDNN benchmark、JIT 編譯、CUDA context 初始化，含進來會嚴重低估真實效能。

要做公平比較，跑前先進 MAXN 模式並鎖頻：
```bash
sudo nvpmodel -m 0   # 全核心+全 GPU 頻率
sudo jetson_clocks   # 鎖最高頻，禁用 DVFS
```
不鎖頻的話 Jetson 會依照負載降頻省電，benchmark 數字會抖。

---

## 效能與精度結果

實測 (Jetson Orin Nano 8GB / 640×640 / `--warmup 15-20 --iters 100-200`)：

#### YOLOv11n（2.6M params, 5.4 MB .pt）

| 格式 | 平均延遲 | p99 | FPS | 加速比 | 模型大小 |
|------|---------|-----|-----|--------|---------|
| PyTorch .pt (FP32) | 35.8 ms | 37.6 ms | **27.9** | 1.00× | 5.4 MB |
| TensorRT .engine (FP16) | 20.8 ms | 21.2 ms | **48.1** | 1.72× | 8.3 MB |
| TensorRT .engine (INT8) | 18.4 ms | 19.6 ms | **54.5** | 1.95× | 5.4 MB |

#### YOLOv11s（9.4M params, 18.4 MB .pt）

| 格式 | 平均延遲 | p99 | FPS | 加速比 | 模型大小 |
|------|---------|-----|-----|--------|---------|
| PyTorch .pt (FP32) | 33.3 ms | 38.9 ms | **30.0** | 1.00× | 18.4 MB |
| TensorRT .engine (FP16) | 23.6 ms | 28.6 ms | **42.4** | 1.41× | 21.4 MB |
| TensorRT .engine (INT8) | 22.1 ms | 23.2 ms | **45.3** | 1.51× | 12.1 MB |

#### YOLOv11m（20.1M params, 38.8 MB .pt）

| 格式 | 平均延遲 | p99 | FPS | 加速比 | 模型大小 |
|------|---------|-----|-----|--------|---------|
| PyTorch .pt (FP32) | 55.5 ms | 56.1 ms | **18.0** | 1.00× | 38.8 MB |
| TensorRT .engine (FP16) | 24.2 ms | 32.3 ms | **41.3** | 2.30× | 42.0 MB |
| TensorRT .engine (INT8) | 24.6 ms | 35.1 ms | **40.6** | 2.25× | 23.2 MB |

**為什麼 FP16 engine 比 .pt 大？** TRT 把模型展開、合併層、加上 metadata，會比原始權重大一點。INT8 因為權重本身變 1/4 又縮回到接近原 .pt 大小。

**精度** (bus.jpg，肉眼判定 5 個物件)：
- FP32：5/5 ✓
- FP16：4/5（漏掉信心 0.62、嚴重遮蔽的 person）
- INT8：4/5（同 FP16，沒有額外掉）

### 反直覺發現

直覺以為「模型越大、量化加速越多」，**實測完全相反**，而且是漂亮的單調遞減：

| 模型 | params | INT8 vs FP16 增益 |
|------|--------|------------------|
| YOLOv11n | 2.6M | **+13%** (48→55 FPS) |
| YOLOv11s | 9.4M | **+7%**  (42→45 FPS) |
| YOLOv11m | 20.1M | **−2%**（在 noise 內，等於沒效果） |

可能解釋：
- 大模型在 FP16 下**已經餵飽 Tensor Core**，INT8 主要省記憶體頻寬，但這部分早已被 TRT 的層融合處理掉了
- 小模型的 wall-clock 含較多固定 overhead（Python 包裝、preprocess、NMS），INT8 縮短的 GPU 段時間反而在 wall-clock 上比例變大
- INT8 本身有 Q/DQ 節點的額外開銷，當 matmul 加速沒拉開時這個 overhead 反咬一口

但 **.pt → FP16 的加速比剛好相反，越大越明顯**：

| 模型 | .pt → FP16 加速 |
|------|----------------|
| YOLOv11n | 1.72× |
| YOLOv11s | 1.41× |
| YOLOv11m | **2.30×** |

yolo11m FP32 跑得很慢（55 ms / 18 FPS），因為 PyTorch 沒對它做太多 kernel 優化；一進 TRT 就拉到 41 FPS。換句話說：**模型越大，TRT 越值得，但 INT8 量化越不值得**。

另一個值得注意的點：**yolo11m FP16（41 FPS）≈ yolo11s INT8（45 FPS）≈ yolo11n FP16（48 FPS）** — 在 Orin Nano 上，這三個組合 FPS 接近，但模型容量差到 8 倍。如果你的場景需要偵測小物件或困難場景，**直接上 yolo11m FP16** 是合理的，沒必要為了快幾 FPS 走 INT8。

不過 yolo11s INT8 的 p99 延遲（23.2 ms）比 FP16 版本（28.6 ms）穩很多 — 量化能降低延遲變異，這在固定 frame budget 的應用（如 30 FPS 視訊串流）很有用。yolo11m INT8 反而 p99 比 FP16 還大（35 vs 32），這個 model size 上 INT8 沒帶來穩定性收益。

### 怎麼選 size + 精度

| 場景 | 建議 |
|------|------|
| 追極限 FPS（不在乎精度幾%） | **yolo11n INT8**（55 FPS, 5.4 MB） |
| 精度優先、可接受 30 FPS | **yolo11n FP32 .pt**（28 FPS, 不用 TRT 匯出，最準） |
| 平衡點，部署首選 | **yolo11n FP16 engine**（48 FPS, 精度幾乎沒掉） |
| 偵測小物件 / 困難場景 | **yolo11m FP16**（41 FPS, 模型容量 8× nano） |
| 即時 30 FPS 串流要穩定 | **yolo11s INT8**（p99 23 ms，最不抖） |
| 想跑 yolo11m 但要省記憶體 | **yolo11m INT8**（23 MB engine, FPS 與 FP16 持平） |

---

## 疑難排解 / 踩雷紀錄

### `torch.cuda.is_available()` 是 False
你裝到了 PyPI 的 CPU wheel。`pip install torch` 在 aarch64 上會抓 CPU-only 版本，覆蓋掉 Jetson 的 CUDA torch。解法：
```bash
pip uninstall -y torch torchvision
pip install --user https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install --user --no-deps torchvision==0.20.0
```

### `libcusparseLt.so.0: cannot open shared object file`
Jetson torch 需要 cuSPARSELt，但本機原本沒裝完整。如果用本機的 user-level torch 2.8 就沒這問題。

### `pypi.jetson-ai-lab.dev` 連不到
這個鏡像已經下線 (NXDOMAIN)。改用 NVIDIA 官方 URL（見上）。

### `python3.10-venv` 缺失，無法建 venv
```bash
pip install --user virtualenv
python3.10 -m virtualenv --system-site-packages .venv
```
或者**不要建 venv** — 直接用 `pip install --user` 安裝到 `~/.local`，這也是本專案 setup.sh 的做法。

### `No module named 'onnx'` 匯出 TensorRT 失敗
ultralytics 走 PyTorch → ONNX → TRT 兩階段。補裝：
```bash
pip install --user onnx onnxslim onnxruntime
```

### TRT 匯出超慢（10+ 分鐘）
**正常**。Auto-tuning 在試上百種 kernel 組合，會記在 cache 裡，下次同條件匯出快很多。INT8 因為要跑校準會再多幾分鐘。

### 找不到 `/dev/video*`
USB webcam 沒接 / 沒被識別。`dmesg | tail` 看看插入時有沒有 log。CSI 相機是用 `nvarguscamerasrc` 走 GStreamer，不會出現在 `/dev/video*`。

### 即時推論卡頓、FPS 忽高忽低
1. 沒鎖頻：`sudo jetson_clocks`
2. 沒進 MAXN：`sudo nvpmodel -m 0`
3. 過熱降頻：`tegrastats` 看 CPU/GPU 溫度，>85°C 會降
4. 解析度太高：相機可能在跑 1080p 但被縮到 640，浪費頻寬。改用 720p 或讓相機直出 640

---

## 進階主題

### Fine-tune 自訂類別

如果 COCO 的 80 類沒有你要的東西（例如「特定品牌的瓶子」），需要：
1. 用 [Roboflow](https://roboflow.com/) 或 [LabelImg](https://github.com/HumanSignal/labelImg) 標註資料集（YOLO 格式：`xywhn` 歸一化）
2. 寫 `data.yaml` 定義類別名稱 + 路徑
3. 訓練（範例）：
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   model.train(data="data.yaml", epochs=100, imgsz=640, device=0)
   ```
4. 把訓練好的 `best.pt` 重新匯出 TensorRT engine

500-1000 張標註好的影像通常夠 fine-tune 一個能用的物件偵測器。

### 換更大的模型

YOLOv11 有 n/s/m/l/x 五種尺寸：
```python
YOLO("yolo11s.pt")  # 9.4M params, ~3.6× yolo11n
YOLO("yolo11m.pt")  # 20.1M params
```
本機實測 `yolo11s` INT8 跑 45 FPS、`yolo11m` FP16 跑 41 FPS（見上面效能表）。`yolo11l/x` 在 Orin Nano 8GB 會吃力，建議跳過。

注意「**換更大模型不等於更好結果**」 — 看你的瓶頸：
- 漏偵測（recall 低）→ 換大模型可能改善（模型容量更大）
- 框不準（bbox 偏移）→ 換大模型常常不改善，要 fine-tune
- 邊緣信心低 → 通常是 domain shift，要 fine-tune 而非換 size

### 多路串流 / 串接 DeepStream

如果要同時跑 4-8 路相機 + 追蹤 (tracking)，純 Python 會卡。NVIDIA DeepStream 是針對這場景的 SDK，內建 GStreamer pipeline + TensorRT 後端 + 多種 tracker。學習曲線比較陡，但 Jetson 是它主場。

### 量化感知訓練 (QAT)

如果 PTQ（post-training quantization，本專案用的）掉精度太多，可以做 QAT：在訓練時就模擬 INT8 的精度損失，讓模型學會適應。需要重新訓練，但 INT8 精度可逼近 FP16。

---

## 專案結構

```
ml-vision/
├── README.md                 # 本指南
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

## 常用診斷指令

```bash
tegrastats                    # GPU/CPU/記憶體/溫度即時狀態
ls /dev/video*                # 確認 USB 相機掛載
gst-launch-1.0 nvarguscamerasrc ! fakesink   # 測 CSI MIPI 相機
sudo nvpmodel -q              # 查目前 power mode
nvidia-smi                    # 確認驅動正常（記憶體不準，看 tegrastats）
```

## 延伸閱讀

- [Ultralytics YOLO 文件](https://docs.ultralytics.com/) — 訓練 / 匯出 / API 完整參考
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/) — 各種 Jetson 上的 AI 範例
- [TensorRT 開發者指南](https://docs.nvidia.com/deeplearning/tensorrt/) — 深入優化技巧
- [YOLO 系列演進論文集](https://github.com/ultralytics/ultralytics#documentation)
- [COCO 資料集](https://cocodataset.org/) — 物件偵測標竿
