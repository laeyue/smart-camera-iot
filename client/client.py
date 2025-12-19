import asyncio
import base64
import json
import signal
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import websockets
from ultralytics import YOLO

# ==== CONFIGURATION ====
SERVER_HOST = "YOUR_SERVER_IP_OR_HOSTNAME"  # e.g. "192.168.1.10"
SERVER_PORT = 7777
WS_PATH = "/ws/pi"
MODEL_PATH = "yolov8n.pt"
TARGET_CLASS_ID = 24  # Backpack in COCO
JPEG_QUALITY = 60
MAX_FPS = 10  # limit to reduce CPU/network
# ========================


def normalize_boxes(
    frame_shape: Tuple[int, int, int],
    boxes_xyxy: List[Tuple[float, float, float, float]],
    confs: List[float],
) -> List[List[float]]:
    """
    Convert absolute xyxy boxes to normalized [x, y, w, h, conf].
    All coordinates are in [0,1] relative to image size.
    """
    h, w = frame_shape[:2]
    norm_boxes: List[List[float]] = []
    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
        bw = x2 - x1
        bh = y2 - y1
        nx = x1 / w
        ny = y1 / h
        nw = bw / w
        nh = bh / h
        norm_boxes.append([nx, ny, nw, nh, float(conf)])
    return norm_boxes


async def send_frames() -> None:
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    uri = f"ws://{SERVER_HOST}:{SERVER_PORT}{WS_PATH}"
    print(f"Connecting to WebSocket server at {uri}")

    last_frame_time = 0.0

    try:
        while True:
            try:
                async with websockets.connect(
                    uri,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=4 * 1024 * 1024,
                ) as ws:
                    print("Connected to server.")
                    while True:
                        # FPS limiting
                        now = time.time()
                        if now - last_frame_time < 1.0 / MAX_FPS:
                            await asyncio.sleep(0.001)
                            continue
                        last_frame_time = now

                        ret, frame = cap.read()
                        if not ret:
                            print("Warning: Failed to read frame from camera.")
                            await asyncio.sleep(0.1)
                            continue

                        # Run YOLO inference for backpack class only
                        results = model(frame, classes=[TARGET_CLASS_ID], verbose=False)[0]

                        boxes_xyxy: List[Tuple[float, float, float, float]] = []
                        confs: List[float] = []
                        if results.boxes is not None and len(results.boxes) > 0:
                            for b in results.boxes:
                                x1, y1, x2, y2 = b.xyxy[0].tolist()
                                conf = float(b.conf[0])
                                boxes_xyxy.append((x1, y1, x2, y2))
                                confs.append(conf)

                        norm_boxes = normalize_boxes(frame.shape, boxes_xyxy, confs)

                        # Encode JPEG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        success, buffer = cv2.imencode(".jpg", frame, encode_param)
                        if not success:
                            print("Warning: JPEG encoding failed.")
                            continue

                        jpg_bytes = buffer.tobytes()
                        b64_image = base64.b64encode(jpg_bytes).decode("ascii")

                        payload = {
                            "image": b64_image,
                            "boxes": norm_boxes,
                        }
                        payload_str = json.dumps(payload)
                        await ws.send(payload_str)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.InvalidURI,
            ) as exc:
                print(f"WebSocket connection error: {exc}. Reconnecting in 3 seconds...")
                await asyncio.sleep(3)
            except OSError as exc:
                print(f"OS error: {exc}. Reconnecting in 3 seconds...")
                await asyncio.sleep(3)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGINT/SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            # Windows / some environments don't support add_signal_handler
            pass

    try:
        loop.run_until_complete(send_frames())
    finally:
        loop.close()


if __name__ == "__main__":
    if SERVER_HOST == "YOUR_SERVER_IP_OR_HOSTNAME":
        print(
            "Please set SERVER_HOST at the top of client.py "
            "to your FastAPI server's IP or hostname."
        )
        sys.exit(1)
    main()


