from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pathlib import Path
import cv2

# ── Download / load model ──────────────────────────────────────────────
model_path = hf_hub_download(
    repo_id="Anzhc/Race-Classification-FairFace-YOLOv8",
    filename="Race-CLS-FairFace_yolov8s.pt"
)
model = YOLO(model_path)

# ── Load OpenCV face detector ──────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Folders ────────────────────────────────────────────────────────────
image_dir = Path("images")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# ── Preprocessing ──────────────────────────────────────────────────────
def preprocess(img, alpha=2, beta=10):
    # alpha = contrast (1.0 is unchanged), beta = brightness (0 is unchanged)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def detect_and_classify(image_path, padding=0.25):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        return

    img = preprocess(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        print(f"{image_path.name}: no faces detected")
        return

    results_out = []

    for i, (x, y, fw, fh) in enumerate(faces):
        # Add padding around the face
        pad_x = int(fw * padding)
        pad_y = int(fh * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        face_crop = img[y1:y2, x1:x2]

        # Pipe into YOLO classifier
        results = model(face_crop, verbose=False)
        r = results[0]
        label = model.names[r.probs.top1]
        conf = float(r.probs.top1conf)

        results_out.append((label, conf))
        print(f"{image_path.name} face {i+1}: {label} ({conf:.2f})")

        # Save cropped face to output folder
        crop_path = output_dir / f"{image_path.stem}_face{i+1}.jpg"
        cv2.imwrite(str(crop_path), face_crop)

    # Write results to .txt sidecar in output folder
    txt_path = output_dir / image_path.with_suffix(".txt").name
    with open(txt_path, "a") as f:
        for label, conf in results_out:
            f.write(f"{label} {conf:.4f}\n")

# ── Run on images folder ───────────────────────────────────────────────
image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
images = [p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions]
print(f"Found {len(images)} images\n")

for img_path in images:
    detect_and_classify(img_path)

print("\nDone!")