# Project Documentation

## What This Project Does Now

This project performs face-based race classification using:

- OpenCV Haar Cascade for face detection
- A YOLOv8 classification model from Hugging Face for per-face classification

It supports two runtime modes:

1. `images` mode: process files from `images/`
2. `live` mode: process a device camera feed and sample frames at a configurable interval (default every `0.5` seconds)

## Repository Files

### `run.py`

This is the main script and now contains a CLI entry point (`main()`), reusable detection/classification helpers, and both `images` and `live` execution modes.

#### Core startup behavior

On startup, the script:

1. Downloads `Race-CLS-FairFace_yolov8s.pt` from Hugging Face repo `Anzhc/Race-Classification-FairFace-YOLOv8`
2. Loads the model with `ultralytics.YOLO`
3. Loads OpenCV frontal-face Haar Cascade (`haarcascade_frontalface_default.xml`)
4. Ensures `output/` exists

#### Main helper functions

- `preprocess(img, alpha=2, beta=10)`
  - Applies `cv2.convertScaleAbs` for contrast/brightness normalization.
- `detect_faces(img)`
  - Converts to grayscale and runs Haar Cascade detection with:
    - `scaleFactor=1.1`
    - `minNeighbors=5`
    - `minSize=(40, 40)`
- `classify_faces(img, faces, source_stem, source_label, padding=0.25)`
  - Crops each detected face (with configurable padding),
  - Classifies each crop with YOLO,
  - Prints per-face label + confidence,
  - Saves each face crop to `output/<source_stem>_faceN.jpg`,
  - Appends predictions to `output/<source_stem>.txt`.

#### Images mode

- Entry: `run_images_mode(padding=0.25)`
- Input folder: `images/` (non-recursive)
- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`
- For each image:
  - read -> preprocess -> detect faces -> classify faces -> save outputs
- If no face is found, it prints `<filename>: no faces detected`.

#### Live mode

- Entry: `run_live_mode(camera_index=0, sample_interval=0.5, padding=0.25, display=True)`
- Opens a device camera via `cv2.VideoCapture(camera_index)`.
- Reads frames continuously, but performs detect+classify only when the sample interval elapses.
- Default interval is `0.5s` (configurable).
- For sampled frames:
  - preprocess -> detect faces
  - if faces exist: classify and write outputs
  - if no faces: print `frame <index>: no faces detected`
- Preview window is shown by default in live mode.
  - Press `q` to quit.
  - `--no-display` disables window preview.

### `setup.py`

This file is currently a helper script, not a packaging setup.

- Downloads the same model from Hugging Face
- Prints downloaded path

### `README.md`

Currently contains only the project title line and no usage instructions.

## CLI Usage (`run.py`)

### Default (images mode)

```bash
python3 run.py
```

### Live mode using default camera (`0`) and default interval (`0.5s`)

```bash
python3 run.py --mode live
```

### Live mode with custom interval

```bash
python3 run.py --mode live --sample-interval 1.0 --camera-index 0
```

### Live mode with another device camera

```bash
python3 run.py --mode live --camera-index 1
```

### Other options

- `--padding <float>`: face crop padding ratio (default `0.25`)
- `--no-display`: disable live preview window in live mode

## Runtime Inputs and Outputs

### Inputs

- Images mode: files in `images/`
- Live mode: device camera index via `--camera-index`
- Network access needed initially to download model weights (unless cached)

### Outputs (always in `output/`)

- Face crops: `<source_stem>_face1.jpg`, `<source_stem>_face2.jpg`, ...
- Prediction text files: `<source_stem>.txt`
  - Each line: `<label> <confidence>`

Notes:

- Text output files are appended to (`"a"` mode), so repeated runs can accumulate duplicate lines unless cleaned.
- In live mode, each sampled frame with faces gets a unique `source_stem` (based on frame index + timestamp), so crops/results are separated by frame.

## Dependencies Used by Code

- `ultralytics`
- `huggingface_hub`
- `opencv-python`

## Current Limitations / Notes

- Detection uses Haar Cascades, which may be less robust than modern deep detectors under difficult angles/lighting.
- No explicit retry/fallback logic for model download/network issues.
- Images mode is non-recursive (`images/` only, no nested folders).
