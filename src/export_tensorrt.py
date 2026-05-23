"""把 .pt 權重匯出成 Jetson TensorRT engine。

用法:
    python src/export_tensorrt.py                              # yolo11n.pt → yolo11n.engine (FP16)
    python src/export_tensorrt.py --weights models/yolo11s.pt --imgsz 640
    python src/export_tensorrt.py --int8 --data coco128.yaml   # INT8 量化 (需校準集)

注意:
- 匯出在 Jetson 上要跑數分鐘到十幾分鐘 (TensorRT 會做 kernel auto-tuning)。
- 產出的 .engine 跟硬體+TRT 版本綁定，換機器要重新匯出。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO .pt to TensorRT .engine for Jetson")
    p.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "yolo11n.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", default=True, help="FP16 (預設開, Jetson Orin 支援)")
    p.add_argument("--int8", action="store_true", help="INT8 量化 (需要 --data 校準集)")
    p.add_argument("--data", default="", help="INT8 校準用 dataset YAML (例: coco128.yaml)")
    p.add_argument("--workspace", type=int, default=4, help="TRT workspace (GB)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.int8 and not args.data:
        raise SystemExit("INT8 量化需要 --data 指定校準資料集 YAML")

    print(f"[i] 載入 {args.weights} ...")
    model = YOLO(args.weights)

    print(f"[i] 匯出 TensorRT engine (imgsz={args.imgsz}, "
          f"{'INT8' if args.int8 else 'FP16' if args.half else 'FP32'}) ...")
    out = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=not args.int8 and args.half,
        int8=args.int8,
        data=args.data or None,
        workspace=args.workspace,
        device=0,
    )
    print(f"[✓] 匯出完成: {out}")


if __name__ == "__main__":
    main()
