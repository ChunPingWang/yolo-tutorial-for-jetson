"""量測 YOLO 在 Jetson 上的推論延遲與 FPS。

用法:
    python src/benchmark.py                                       # 預設權重, 200 次, 640x640
    python src/benchmark.py --weights models/yolo11n.engine       # TensorRT engine
    python src/benchmark.py --warmup 30 --iters 500
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO benchmark on Jetson")
    p.add_argument("--weights", default=str(PROJECT_ROOT / "models" / "yolo11n.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    # 隨機影像，避免測到磁碟 IO 與解碼
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    print(f"[i] weights={args.weights}  imgsz={args.imgsz}  device={args.device}")
    print(f"[i] warmup={args.warmup} iters={args.iters}")

    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)

    latencies_ms: list[float] = []
    t_start = time.perf_counter()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    total_s = time.perf_counter() - t_start

    avg = statistics.mean(latencies_ms)
    p50 = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=20)[18]  # 95th
    p99 = statistics.quantiles(latencies_ms, n=100)[98]
    fps = args.iters / total_s

    print()
    print(f"  推論延遲 (ms)   平均 {avg:6.2f}   p50 {p50:6.2f}   p95 {p95:6.2f}   p99 {p99:6.2f}")
    print(f"  輸送量          {fps:6.1f} FPS")
    print(f"  總耗時          {total_s:6.2f} s  ({args.iters} iters)")


if __name__ == "__main__":
    main()
