import argparse
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

import threading
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from rembg import remove, new_session

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

_session = new_session()

# ── Constants ─────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOUND_CONFIDENCE_THRESHOLD = 0.50

CLASS_TO_TRACK = {
    "Black": "black.mp3",
    "East Asian": "East_Asian.mp3",
    "Indian": "indian.mp3",
    "Latino_Hispanic": "Latino_Hispanic.mp3",
    "Middle Eastern": "Middle Eastern.mp3",
    "Southeast Asian": "Southeast Asian.mp3",
    "White": "white.mp3",
}

# ── Directories ───────────────────────────────────────────────────────
IMAGE_DIR = Path("images")
OUTPUT_DIR = Path("output")
MUSIC_DIR = Path("music")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Model + face detector ─────────────────────────────────────────────
model_path = hf_hub_download(
    repo_id="Anzhc/Race-Classification-FairFace-YOLOv8",
    filename="Race-CLS-FairFace_yolov8s.pt",
)
model = YOLO(model_path)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Audio state ───────────────────────────────────────────────────────
_sound_process = None


def is_sound_playing():
    return _sound_process is not None and _sound_process.poll() is None


def _resolve_track(filename):
    path = MUSIC_DIR / filename
    if path.exists():
        return path

    target = filename.lower()
    for candidate in MUSIC_DIR.glob("*.mp3"):
        if candidate.name.lower() == target:
            return candidate
    return None


def play_sound(label):
    global _sound_process

    if is_sound_playing() or shutil.which("afplay") is None:
        return

    track_name = CLASS_TO_TRACK.get(label)
    track = _resolve_track(track_name) if track_name else None

    if track is None:
        all_tracks = list(MUSIC_DIR.glob("*.mp3"))
        if not all_tracks:
            return
        track = random.choice(all_tracks)

    try:
        _sound_process = subprocess.Popen(
            ["afplay", str(track)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Playing sound: {track.name}")
    except OSError as exc:
        print(f"Could not play sound {track.name}: {exc}")


# ── Image helpers ─────────────────────────────────────────────────────
def preprocess(img, alpha=2, beta=10):
    # Remove background (returns BGRA image with transparent background)
    no_bg = remove(img, session=_session)

    # Composite onto a white background so convertScaleAbs works correctly
    if no_bg.shape[2] == 4:
        alpha_mask = no_bg[:, :, 3:4] / 255.0
        bgr = no_bg[:, :, :3].astype(np.float32)
        white_bg = np.ones_like(bgr) * 255
        img = (bgr * alpha_mask + white_bg * (1 - alpha_mask)).astype(np.uint8)
    else:
        img = no_bg

    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )


def crop_face(img, x, y, fw, fh, padding=0.25):
    h, w = img.shape[:2]
    pad_x, pad_y = int(fw * padding), int(fh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_y)
    return img[y1:y2, x1:x2]


# ── Core detect + classify ───────────────────────────────────────────
def detect_and_classify(img, source_tag, padding=0.25):
    faces = detect_faces(img)
    if len(faces) == 0:
        print(f"{source_tag}: no faces detected")
        return

    results = []

    for i, (x, y, fw, fh) in enumerate(faces):
        face_img = crop_face(img, x, y, fw, fh, padding)

        r = model(face_img, verbose=False)[0]
        label = model.names[r.probs.top1]
        conf = float(r.probs.top1conf)
        results.append((label, conf))

        print(f"{source_tag} face {i + 1}: {label} ({conf:.2f})")
        print(f"Label detected: {label}")

        if conf > SOUND_CONFIDENCE_THRESHOLD:
            play_sound(label)

        crop_path = OUTPUT_DIR / f"{source_tag}_face{i + 1}.jpg"
        cv2.imwrite(str(crop_path), face_img)

    txt_path = OUTPUT_DIR / f"{source_tag}.txt"
    with open(txt_path, "a", encoding="utf-8") as f:
        for label, conf in results:
            f.write(f"{label} {conf:.4f}\n")


# ── Mode runners ─────────────────────────────────────────────────────
def run_images_mode(padding=0.25):
    images = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Found {len(images)} images\n")

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not read {path}")
            continue
        detect_and_classify(preprocess(img), source_tag=path.stem, padding=padding)

    print("\nDone!")


def run_live_mode(camera_index=0, sample_interval=0.5, padding=0.25, display=True):
    if sample_interval <= 0:
        raise ValueError("sample_interval must be > 0")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index: {camera_index}")
        return

    print(
        f"Live mode started (camera={camera_index}, interval={sample_interval}s). "
        "Press 'q' to quit."
    )

    frame_index = 0
    next_sample_time = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame. Stopping.")
                break

            frame_index += 1
            now = time.monotonic()

            if now >= next_sample_time:
                next_sample_time = now + sample_interval

                if is_sound_playing():
                    print(f"frame {frame_index}: detection paused (sound playing)")
                else:
                    tag = f"live_frame{frame_index}_{int(time.time() * 1000)}"
                    detect_and_classify(frame, source_tag=tag, padding=padding)

            if display:
                cv2.imshow("Live Face Detection (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

    print("Live mode stopped.")

# --- Shared state ---
_latest_frame = None
_frame_lock = threading.Lock()

flask_app = Flask(__name__)
CORS(flask_app, origins="*")

@flask_app.route("/frame", methods=["POST", "OPTIONS"])
def receive_frame():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response, 200

    global _latest_frame
    data = request.get_json(force=True)
    b64 = data.get("frame", "")
    if not b64:
        return jsonify({"error": "no frame"}), 400

    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "decode failed"}), 400

    with _frame_lock:
        _latest_frame = frame

    response = jsonify({"ok": True})
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


def start_frame_server(host="0.0.0.0", port=5050):
    """Start the Flask receiver in a daemon thread."""
    t = threading.Thread(
        target=lambda: flask_app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"Frame server listening on {host}:{port}")


def run_live_mode_webapp(
    sample_interval=0.5,
    padding=0.25,
    display=True,
    server_host="0.0.0.0",
    server_port=5050,
):
    if sample_interval <= 0:
        raise ValueError("sample_interval must be > 0")

    start_frame_server(host=server_host, port=server_port)
    print(f"Live mode started (webapp feed, interval={sample_interval}s). Press 'q' to quit.")

    frame_index = 0
    next_sample_time = time.monotonic()

    try:
        while True:
            with _frame_lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None

            if frame is None:
                time.sleep(0.05)
                continue

            frame_index += 1
            now = time.monotonic()

            if now >= next_sample_time:
                next_sample_time = now + sample_interval

                if is_sound_playing():
                    print(f"frame {frame_index}: detection paused (sound playing)")
                else:
                    tag = f"webapp_frame{frame_index}_{int(time.time() * 1000)}"
                    detect_and_classify(preprocess(frame), source_tag=tag, padding=padding)

            if display:
                cv2.imshow("Live Face Detection (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if display:
            cv2.destroyAllWindows()

    print("Live mode stopped.")


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Face detection + race classification for images or live video."
    )
    parser.add_argument(
        "--mode", choices=["images", "live", "web"], default="images",
        help="Run image-folder processing or live video processing.",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=0.5,
        help="Seconds between samples in live mode.",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="Device camera index for live mode.",
    )
    parser.add_argument(
        "--padding", type=float, default=0.25,
        help="Padding ratio around detected face crop.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live preview window.",
    )
    args = parser.parse_args()

    if args.mode == "live":
        run_live_mode(
            camera_index=args.camera_index,
            sample_interval=args.sample_interval,
            padding=args.padding,
            display=not args.no_display,
        )
    elif args.mode == "web":
        run_live_mode_webapp(
            sample_interval=0.5,
            padding=0.25,
            display=True,
            server_host="0.0.0.0",
            server_port=5050,
        )
    else:
        run_images_mode(padding=args.padding)


if __name__ == "__main__":
    main()
