"""Jetson 即時相機 YOLO 偵測 (USB / CSI 自動偵測)。

用法:
    python src/detect_camera.py                          # 自動找相機
    python src/detect_camera.py --source 0               # /dev/video0
    python src/detect_camera.py --source csi --csi-id 0  # CSI MIPI 相機
    python src/detect_camera.py --weights models/yolo11n.engine --headless --save output/run.mp4

按 q 離開。--headless 時無視窗，可用於無顯示器跑。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time YOLO detection on Jetson camera")
    p.add_argument("--source", default="auto",
                   help="auto | csi | 整數 (/dev/videoN) | 影片路徑")
    p.add_argument("--csi-id", type=int, default=0, help="CSI sensor id (僅 source=csi)")
    p.add_argument("--csi-width", type=int, default=1280)
    p.add_argument("--csi-height", type=int, default=720)
    p.add_argument("--csi-fps", type=int, default=30)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help=".pt 或 .engine")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="0=GPU / cpu")
    p.add_argument("--headless", action="store_true", help="不開視窗 (無顯示器)")
    p.add_argument("--save", default="", help="存成影片，例: output/run.mp4")
    p.add_argument("--max-frames", type=int, default=0, help="跑 N 幀後退出 (0=無限)")
    return p.parse_args()


def gstreamer_csi_pipeline(sensor_id: int, w: int, h: int, fps: int) -> str:
    """Argus → BGR appsink，給 Jetson CSI MIPI 相機 (IMX219 / IMX477 等)。"""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, format=NV12, framerate={fps}/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true sync=false max-buffers=2"
    )


def open_capture(args: argparse.Namespace) -> tuple[cv2.VideoCapture, str]:
    """回傳 (cap, 來源描述)；找不到就 raise。"""
    src = args.source

    # 顯式 CSI
    if src == "csi":
        pipe = gstreamer_csi_pipeline(args.csi_id, args.csi_width, args.csi_height, args.csi_fps)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, f"CSI sensor-id={args.csi_id}"
        raise RuntimeError("無法開啟 CSI 相機 — OpenCV 是否含 GStreamer 支援？")

    # 整數 → /dev/videoN
    if src.isdigit():
        idx = int(src)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap, f"/dev/video{idx}"
        raise RuntimeError(f"無法開啟 /dev/video{idx}")

    # 自動
    if src == "auto":
        # 先試 USB
        for idx in range(0, 5):
            if os.path.exists(f"/dev/video{idx}"):
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    return cap, f"/dev/video{idx} (auto)"
                cap.release()
        # 再試 CSI
        pipe = gstreamer_csi_pipeline(0, args.csi_width, args.csi_height, args.csi_fps)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "CSI sensor-id=0 (auto)"
        raise RuntimeError(
            "找不到任何相機。請接 USB webcam 或 CSI MIPI 相機。\n"
            "提示：先用 `ls /dev/video*` 確認；CSI 用 `gst-launch-1.0 nvarguscamerasrc ! fakesink` 測試。"
        )

    # 影片路徑
    cap = cv2.VideoCapture(src)
    if cap.isOpened():
        return cap, src
    raise RuntimeError(f"無法開啟來源: {src}")


def main() -> int:
    args = parse_args()
    cap, label = open_capture(args)
    print(f"[i] 來源: {label}")

    model = YOLO(args.weights)
    names = model.names

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        # 解析度待第一幀拿到再開 writer
        save_path = args.save
    else:
        save_path = ""

    fps_window: deque[float] = deque(maxlen=30)
    last_t = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[!] 讀取結束或失敗")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot()

            # FPS
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            fps_window.append(1.0 / dt if dt > 0 else 0.0)
            fps = sum(fps_window) / len(fps_window)

            n_det = len(results[0].boxes)
            cv2.putText(annotated, f"FPS {fps:5.1f}  det {n_det}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if save_path:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))
                writer.write(annotated)

            if not args.headless:
                cv2.imshow("ml-vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[✓] 影片已存到: {save_path}")
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"[✓] 處理 {frame_idx} 幀，平均 FPS {sum(fps_window)/max(len(fps_window),1):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
