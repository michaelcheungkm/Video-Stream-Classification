# Video-Stream-Classification

Face detection + race classification using OpenCV (Haar Cascade) and a YOLOv8 classifier from Hugging Face.

## Prerequisites

- Python 3.9+ (recommended)
- Webcam (for live mode) or input images in `images/`

## Setup (with venv)

### 1) Create a virtual environment

```bash
python3 -m venv .venv
```

### 2) Activate it

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

### Images mode (default)

Put images in the `images/` folder and run:

```bash
python3 run.py
```

### Live mode (default webcam, every 0.5s)

```bash
python3 run.py --mode live
```

### Live mode with custom sampling interval

```bash
python3 run.py --mode live --sample-interval 1.0 --camera-index 0
```

### Live mode with another device camera

```bash
python3 run.py --mode live --camera-index 1
```

### Useful options

- `--padding 0.25` : padding ratio around detected face crop
- `--no-display` : disable preview window in live mode

## Output

All outputs are written to `output/`:

- Cropped face images: `<source>_faceN.jpg`
- Classification text file(s): `<source>.txt`

## Notes

- The model is downloaded from Hugging Face on first run and cached locally.
- Prediction text files are appended to, so repeated runs can add more lines.
- When a face is recognized, one track from `music/` is played (if no other track is currently playing).
- Audio playback uses `afplay` (macOS).
- In live mode, face labeling uses Haar Cascade (`Face`) instead of YOLO race classes.