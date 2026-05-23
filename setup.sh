#!/usr/bin/env bash
# Jetson Orin Nano 環境一鍵建置
# JetPack 6 / CUDA 12.6 / Python 3.10 / TensorRT 10.3
#
# 本機已有 user-level CUDA torch 2.8 + torchvision 0.23 (~/.local/lib/python3.10)，
# 所以不建 venv，直接 pip install --user 補裝 ultralytics 與相關套件。
set -euo pipefail

PY=/usr/bin/python3.10

echo "[1/3] 確認系統 Python 3.10 ..."
$PY --version

echo "[2/3] 安裝 ultralytics / opencv / onnx (user-level, ~/.local) ..."
$PY -m pip install --user --upgrade \
    "numpy>=1.26,<2.0" \
    "ultralytics>=8.3.0" \
    "opencv-python>=4.10.0" \
    "pillow>=10.0.0" \
    "psutil>=5.9.0" \
    onnx onnxslim onnxruntime

echo "[3/3] 驗證 ..."
$PY - <<'PY'
import torch, cv2, numpy as np, ultralytics, tensorrt as trt
print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"opencv      : {cv2.__version__}")
print(f"numpy       : {np.__version__}")
print(f"ultralytics : {ultralytics.__version__}")
print(f"tensorrt    : {trt.__version__}")
PY

echo
echo "完成。直接用 /usr/bin/python3.10 跑 src/*.py 即可。"
echo "（如果 yolo CLI 找不到：export PATH=\"\$HOME/.local/bin:\$PATH\"）"
