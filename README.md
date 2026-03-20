# SAM Web Annotator

Browser-based image annotation tool for instance segmentation with SAM (Segment Anything) or YOLO-seg models.

## Features

- Load local dataset folder.
- Load a segmentation model (SAM checkpoints or YOLO-seg `.pt/.onnx`).
- Auto-segment all objects in the image.
- SAM 2 point prompts (positive/negative clicks) for single-object segmentation.
- Video support (frame extraction, point prompts, optional propagation).
- Keep needed objects, remove the rest.
- Edit polygon boundaries (move vertices, move whole mask).
- Built-in path picker for dataset/model/classes/labels folders.
- Load classes from YAML/TXT or from manual list.
- `Save` current image and `Save All` for all unsaved images.
- Keyboard shortcuts and image switching by arrow keys.
- Zoom in/out/fit and mouse wheel zoom.
- Save in YOLOv8 segmentation TXT format (polygon points).

## Dataset layout

Supported layouts:

- `dataset/images/...` with `dataset/labels/...`
- flat image folder
- custom labels folder path (optional, can be outside dataset)

## Install

```powershell
cd sam_web_annotator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `torch` wheels fail to install, use the official PyTorch install selector for your CUDA/CPU setup.
If `segment-anything` is missing on PyPI in your environment, install it from GitHub:

```powershell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## SAM 2 quick start

1. Install SAM 2 (official repo recommends WSL on Windows):

```powershell
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

2. Download a SAM 2 checkpoint (for example `sam2.1_hiera_large.pt`) from the official list.
3. Use the matching config YAML from the SAM 2 repo, for example `configs/sam2.1/sam2.1_hiera_l.yaml`.
4. In the app UI select `Segment Anything 2 (SAM 2)`, set the checkpoint in “Segmentation model path” and the config in “SAM 2 config YAML”.
5. Click `Auto-segment`.

### Point prompts (SAM 2)

1. Select `Segment Anything 2 (SAM 2)` engine.
2. Load a SAM 2 checkpoint and matching config YAML.
3. Enable `Point mode`, then left-click to add positive points and right-click (or Shift-click) to add negative points.
4. Click `Segment from points` (or press `Enter`) to add the mask.

### Video

1. Use the `Video` section to load an `.mp4/.avi/.mov` file.
2. Frames are extracted into a cache dataset and shown in the main navigator.
3. Use point prompts to segment objects per-frame, or click `Propagate video` to track masks and save labels.

## SAM model

Download a SAM checkpoint (`sam_vit_h_4b8939.pth`, `sam_vit_l_0b3195.pth`, `sam_vit_b_01ec64.pth`) and set the model type
(`vit_h`, `vit_l`, `vit_b`) in the UI. The model path can point to the `.pth` checkpoint file.

## Run

```powershell
cd sam_web_annotator
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Quick use

1. Set dataset folder.
2. Optionally set model path.
3. Optionally set classes YAML/TXT or classes list.
4. Optionally set output labels folder.
5. Click `Load Dataset`.
6. Choose segmentation engine and model path (for SAM 2 also set config YAML).
7. Use `Auto-segment`, keep needed masks, edit, then `Save` or `Save All`.

## Hotkeys

- `Ctrl+S`: save current image
- `Ctrl+Shift+S`: save all unsaved images
- `Ctrl+Z`: undo last mask operation
- `P`: auto-segment current image
- `N`: toggle draw mode (click points, Enter to finish)
- `T`: toggle point mode (SAM 2 prompts)
- `Enter`: apply point prompts when in point mode
- `Left/Right`: previous/next image
- `Delete`: delete selected mask
- `+` / `-`: zoom
- `0`: fit zoom
