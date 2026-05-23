# Backlog

目前已完成五個 YOLOv11 size (n/s/m/l/x) 的 .pt vs TRT FP16 vs INT8 benchmark 與精度比較。本文整理「**接下來還值得做、但需要相機或更多時間**」的實驗與功能。

優先序：P0（接相機後第一個要驗）> P1（核心功能）> P2（優化）> P3（探索）。

---

## A. 相機到位後立刻驗證 (P0)

### A1. dummy data vs 真實相機 FPS 落差
**為什麼重要**：目前所有 benchmark 都餵 `np.random.randint` 隨機影像。真實相機輸入會多出：
- V4L2 / GStreamer 取幀的 I/O 時間
- YUV → BGR 色彩空間轉換
- letterbox 縮放（隨相機解析度而變）

**做法**：USB webcam @ 1280×720 → `detect_camera.py` 跑 3 分鐘，記 FPS。對比 `benchmark.py` 的同 model + 同 imgsz 數字。
**預期**：真實 FPS 比 dummy 低 5-15%，差距越大代表 pipeline 越值得優化。

### A2. CSI MIPI vs USB webcam 比較
**為什麼重要**：CSI 走 `nvarguscamerasrc` (NVIDIA 硬體 ISP)，USB 走 V4L2 (CPU 解碼)。Jetson 主場是 CSI。
**做法**：同一個 model + imgsz，分別接 USB 和 CSI（IMX219/IMX477）。記 FPS + CPU 使用率 (`tegrastats`)。
**預期**：CSI 應該 CPU 使用率低很多，FPS 更穩。

### A3. 端到端延遲（不只推論）
**為什麼重要**：使用者感受的是「動作發生 → 螢幕變化」總延遲，不是模型的 18ms。要量含相機 buffer + 顯示 buffer。
**做法**：拍螢幕上的計時器、相機回拍、視覺化在 OpenCV 視窗，量總時差。或用閃光燈 + photodiode 量更準。
**預期**：60-200 ms 端到端，差很多。

### A4. 解析度對 FPS 的影響
**為什麼重要**：相機可以 1080p / 720p / 480p，pipeline 開銷不同。
**做法**：固定 model = yolo11n-int8.engine，相機輸出依序設 480p / 720p / 1080p，量 FPS。
**預期**：480p 最快但小物件偵測效果差，720p 通常是 sweet spot。

---

## B. 不用相機也能做的（之前沒跑） (P1)

### B1. imgsz 縮小能不能讓 yolo11x INT8 過關？
**為什麼重要**：yolo11x INT8 在 imgsz=640 匯出失敗（記憶體不足）。改 imgsz=320 / 416 可能就過得了。
**做法**：
```bash
python src/export_tensorrt.py --weights models/yolo11x.pt --imgsz 320 --int8 --data coco128.yaml --workspace 1
```
**預期**：320 應該過得了，FPS 可能比 yolo11x@640 FP16 高。

### B2. mAP 真正的精度評估
**為什麼重要**：我們只比過 bus.jpg 一張圖（5 物件）。COCO val 有 5000 張，要看 mAP@50 / mAP@50-95 才知道 INT8 真正掉多少精度。
**做法**：
```bash
yolo val task=detect model=models/yolo11n-int8.engine data=coco.yaml imgsz=640
yolo val task=detect model=models/yolo11n-fp16.engine data=coco.yaml imgsz=640
yolo val task=detect model=models/yolo11n.pt data=coco.yaml imgsz=640
```
（需下載 COCO val 約 800MB）
**預期**：FP16 vs FP32 mAP 差 < 0.5、INT8 差 1-2%。

### B3. MAXN 模式 + 鎖頻對 benchmark 的影響
**為什麼重要**：我們所有 benchmark 都沒鎖頻。Jetson DVFS 會省電降頻，數字可能比實際能達到的低。
**做法**：
```bash
sudo nvpmodel -m 0     # MAXN
sudo jetson_clocks     # 鎖最高頻
python src/benchmark.py --weights models/yolo11n-int8.engine
```
量五個 size，看 FPS 提升幅度。**預期**：FPS 提升 5-20%，p99 變得更穩。

### B4. 多執行緒推論
**為什麼重要**：實際應用常需同時處理多路相機。
**做法**：寫一個 `multi_stream_benchmark.py`，開 2 / 4 / 8 個 thread 同時呼叫同一個 model。看 GPU 是不是被餵飽（單 stream 可能 GPU 還有閒）。
**預期**：2 stream 還能近線性擴展，4+ 就會碰到 GPU 算力上限。

### B5. 對比其他偵測器
**為什麼重要**：YOLO 不是唯一選擇。RT-DETR / YOLO-NAS 在某些場景更準。
**做法**：用 ultralytics 載 `rtdetr-l.pt` / `yolo-nas-s.pt`，跑同樣的 benchmark + bus.jpg 精度。
**預期**：RT-DETR 通常更準但較慢；YOLO-NAS 較舊但量化後在 Jetson 表現好。

---

## C. 「社區住戶安全系統」核心功能 (P1)

### C1. 物件追蹤 (Object Tracking)
**為什麼重要**：偵測只告訴你「有一個人」，追蹤才能說「同一個人從門口走到走廊」。對「徘徊偵測」「重複進出」這類用例必要。
**做法**：ultralytics 內建 ByteTrack / BoT-SORT：
```python
model.track(source="rtsp://...", tracker="bytetrack.yaml", persist=True)
```
**評估**：CPU 使用率、ID switch 數、跨幀 ID 穩定度。

### C2. 進出計數
**用例**：大門進出人數統計、車輛流量。
**做法**：定義一條虛擬線，追蹤 ID 越線觸發 enter/exit 事件。
**雛形腳本**：`src/people_counter.py`（待寫）

### C3. 跌倒偵測 (Fall Detection)
**為什麼重要**：社區老人關懷的標配。
**做法**：兩條路線：
- 簡單：物件偵測 bbox 寬高比 + 持續時間判定（< 0.5 維持 > 2 秒 → 可能跌倒）
- 完整：YOLO-Pose 抽 17 個關鍵點 → 訓練小 classifier
**評估**：精準率（避免誤報）優先於召回率。

### C4. 包裹 / 物品偵測
**用例**：宅配丟包、垃圾未清。
**做法**：fine-tune yolo11n.pt 加「box / package / trash」類別。需要 200-500 張本地拍的標註圖。

### C5. 車牌辨識
**用例**：訪客車輛紀錄、長停車輛告警。
**做法**：兩階段 — YOLO 偵測車牌 → CRNN/PaddleOCR 讀號碼。台灣車牌需要在地化訓練資料。

### C6. 隱私處理（人臉/車牌模糊）
**為什麼重要**：依個資法，存影像要遮蔽可識別資訊。
**做法**：偵測完畫面在存檔前 OpenCV `GaussianBlur` 對應 bbox 區域。臉先用 YOLO-Face 或 MediaPipe 偵測。

### C7. 事件儲存與告警
**為什麼重要**：偵測完要能被人看到 / 收到通知。
**做法**：
- 事件寫入 SQLite（時間、類別、bbox、影像縮圖路徑）
- 觸發條件達成 → Webhook / Line Notify / MQTT
- 簡單 Flask 後台看事件清單與回放

### C8. RTSP 多路串流
**用例**：社區多個出入口的相機統一處理。
**做法**：DeepStream Python bindings 或自寫 OpenCV + 多 thread。

---

## D. 模型優化 (P2)

### D1. Fine-tune 自訂類別
本案場景應該有自己的類別（如「外送員制服」「特定品牌包裹」），COCO 80 類覆蓋不到。流程：
1. 用 Roboflow 標 500-1000 張
2. `model.train(data=...)` 在 Jetson 上跑（n size 大約 1 小時 / 100 epoch）
3. 重新匯出 TensorRT

### D2. 量化感知訓練 (QAT)
本專案用的是 PTQ (Post-Training Quantization)，INT8 會掉 ~1-2% mAP。QAT 在訓練時就模擬 INT8 行為，能拉回到接近 FP16。需要重新訓練。

### D3. 多解析度動態切換
近場景用低解析度衝 FPS，遠場景切高解析度抓細節。需要場景判斷邏輯。

### D4. 模型蒸餾
用 yolo11l 當 teacher 訓練 yolo11n student，可能讓 nano 在自訂資料集上達到接近 l 的精度。

---

## E. 工程化 (P3)

### E1. systemd service 開機自啟
讓系統當機 / 重開機後自動恢復。

### E2. Docker 化
封裝整個推論服務，方便跨機器部署到其他 Jetson。NVIDIA 有 `nvcr.io/nvidia/l4t-jetpack` 基底鏡像。

### E3. 模型版本管理
用 MLflow / DVC 追蹤不同 fine-tune 的權重，避免「哪個 .pt 是最好那一版」的混亂。

### E4. 監控 / 自我健康檢查
推論 FPS 突然掉 50% / GPU 溫度過高 / 相機斷線 → 自動告警。

### E5. OTA 更新
模型權重從 server 推到 Jetson，不用每次都重新登入機器手動換。

---

## 已完成（紀錄參考）

- [x] 五 size YOLOv11 部署：n/s/m/l/x .pt + FP16 engine
- [x] 四 size INT8 engine：n/s/m/l（x 因記憶體匯出失敗）
- [x] 三精度對比 benchmark（含 p50/p95/p99）
- [x] bus.jpg 5 物件精度驗證
- [x] TensorRT workspace 陷阱與解法 (--workspace 2)
- [x] PyPI CPU wheel 陷阱與 Jetson NVIDIA wheel 解法
- [x] README 教學指南（基礎知識 → 動手做 → 進階主題）
