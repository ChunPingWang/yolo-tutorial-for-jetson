"""單張或批次圖檔的 YOLO 物件偵測。

用法:
    python src/detect_image.py --source data/bus.jpg
    python src/detect_image.py --source data/ --conf 0.4 --imgsz 640
    python src/detect_image.py --source https://ultralytics.com/images/bus.jpg
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo11n.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO image detection on Jetson")
    p.add_argument("--source", required=True, help="圖檔、資料夾或 URL")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO 權重 (.pt 或 .engine)")
    p.add_argument("--conf", type=float, default=0.25, help="信心閾值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 閾值")
    p.add_argument("--imgsz", type=int, default=640, help="輸入解析度")
    p.add_argument("--device", default="0", help="0 = GPU, cpu = CPU")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="輸出資料夾")
    p.add_argument("--save-json", action="store_true", help="額外輸出偵測結果 JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights)
    if not weights.exists() and weights.suffix == ".pt":
        # ultralytics 會自動下載，但放到 models/ 比較整齊
        print(f"[i] 權重 {weights} 不存在 — ultralytics 會下載到當前目錄")
    model = YOLO(str(weights))

    t0 = time.perf_counter()
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(output_dir),
        name="detect_image",
        exist_ok=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    n_imgs = len(results)
    total_det = sum(len(r.boxes) for r in results)
    save_dir = Path(results[0].save_dir) if results else output_dir
    print(f"[✓] 推論 {n_imgs} 張圖, 共 {total_det} 個物件, 耗時 {elapsed:.2f}s "
          f"({n_imgs / elapsed:.1f} img/s)")
    print(f"[✓] 標註圖已存到: {save_dir}")

    if args.save_json:
        payload = []
        for r in results:
            payload.append({
                "file": Path(r.path).name,
                "size": [r.orig_shape[1], r.orig_shape[0]],  # [W, H]
                "detections": [
                    {
                        "class": model.names[int(c)],
                        "confidence": float(conf),
                        "bbox_xyxy": [float(x) for x in box],
                    }
                    for box, conf, c in zip(
                        r.boxes.xyxy.cpu().numpy(),
                        r.boxes.conf.cpu().numpy(),
                        r.boxes.cls.cpu().numpy(),
                    )
                ],
            })
        json_path = save_dir / "detections.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[✓] JSON 已存到: {json_path}")


if __name__ == "__main__":
    main()
