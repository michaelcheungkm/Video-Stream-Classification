from huggingface_hub import hf_hub_download

# Choose your size: yolov8n (fastest), yolov8s, or yolov8m (most accurate)
model_path = hf_hub_download(
    repo_id="Anzhc/Race-Classification-FairFace-YOLOv8",
    filename="Race-CLS-FairFace_yolov8s.pt"  # swap n/s/m as needed
)
print(f"Downloaded to: {model_path}")