from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    PointPredictRequest,
    PredictRequest,
    SaveAllRequest,
    SaveAnnotationsRequest,
    Segment,
    SessionRequest,
    VideoPointRequest,
    VideoPropagateRequest,
    VideoSessionRequest,
)
from .storage import (
    discover_images,
    image_absolute_path,
    load_classes,
    read_annotations,
    resolve_path,
    save_annotations,
)
from .segment_inference import SegmentRunner
from .video_storage import VideoInfo, prepare_video_dataset


@dataclass
class SessionState:
    dataset_dir: Path | None = None
    images: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    model_path: Path | None = None
    labels_dir: str | None = None


@dataclass
class VideoState:
    info: VideoInfo | None = None


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="SAM Web Annotator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_state = SessionState()
_video_state = VideoState()
_lock = Lock()
_runner = SegmentRunner()


def _state_snapshot() -> SessionState:
    with _lock:
        return SessionState(
            dataset_dir=_state.dataset_dir,
            images=list(_state.images),
            classes=list(_state.classes),
            model_path=_state.model_path,
            labels_dir=_state.labels_dir,
        )


def _video_snapshot() -> VideoInfo | None:
    with _lock:
        return _video_state.info


def _require_dataset() -> SessionState:
    snapshot = _state_snapshot()
    if snapshot.dataset_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured. Call POST /api/session first.")
    return snapshot


def _require_video() -> VideoInfo:
    info = _video_snapshot()
    if info is None:
        raise HTTPException(status_code=400, detail="Video session is not configured. Call POST /api/video/session first.")
    return info


def _ensure_image_known(snapshot: SessionState, image_path: str) -> None:
    if image_path not in snapshot.images:
        raise HTTPException(status_code=404, detail=f"Image is not in the active dataset: {image_path}")


def _session_payload(snapshot: SessionState) -> dict:
    return {
        "dataset_dir": str(snapshot.dataset_dir) if snapshot.dataset_dir else None,
        "image_count": len(snapshot.images),
        "images": snapshot.images,
        "classes": snapshot.classes,
        "model_path": str(snapshot.model_path) if snapshot.model_path else None,
        "labels_dir": snapshot.labels_dir,
    }


def _list_roots() -> List[str]:
    if os.name == "nt":
        roots: List[str] = []
        for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{drive}:\\")
            if root.exists():
                roots.append(str(root))
        return roots
    return ["/"]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/session")
def get_session() -> dict:
    return _session_payload(_state_snapshot())


@app.post("/api/session")
def set_session(payload: SessionRequest) -> dict:
    dataset_dir = resolve_path(Path.cwd(), payload.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Dataset directory does not exist: {dataset_dir}")

    classes = load_classes(dataset_dir, payload.classes, payload.classes_file)
    images = discover_images(dataset_dir)
    if not images:
        raise HTTPException(status_code=400, detail="No images found in the selected dataset directory.")

    model_path = None
    if payload.model_path:
        model_path = resolve_path(dataset_dir, payload.model_path)

    labels_dir = None
    if payload.labels_dir:
        labels_root = resolve_path(dataset_dir, payload.labels_dir)
        if labels_root.exists() and not labels_root.is_dir():
            raise HTTPException(status_code=400, detail=f"Labels path is not a directory: {labels_root}")
        labels_dir = str(labels_root)

    with _lock:
        _state.dataset_dir = dataset_dir
        _state.images = images
        _state.classes = classes
        _state.model_path = model_path
        _state.labels_dir = labels_dir
        _video_state.info = None

    _runner.reset_video()

    return _session_payload(_state_snapshot())


@app.get("/api/images")
def get_images() -> dict:
    snapshot = _require_dataset()
    return {"images": snapshot.images}


@app.get("/api/image")
def get_image(path: str = Query(..., description="Relative image path inside dataset")) -> FileResponse:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    image_path = image_absolute_path(snapshot.dataset_dir, path)  # type: ignore[arg-type]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
    return FileResponse(image_path)


@app.get("/api/annotations")
def get_annotations(path: str = Query(..., description="Relative image path inside dataset")) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    try:
        segments = read_annotations(snapshot.dataset_dir, path, labels_dir=snapshot.labels_dir)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"segments": [segment.model_dump() for segment in segments]}


@app.post("/api/annotations")
def put_annotations(
    payload: SaveAnnotationsRequest,
    path: str = Query(..., description="Relative image path inside dataset"),
) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    try:
        output_path = save_annotations(
            snapshot.dataset_dir,  # type: ignore[arg-type]
            path,
            payload.segments,
            labels_dir=snapshot.labels_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True, "label_path": str(output_path)}


@app.post("/api/annotations/batch")
def put_annotations_batch(payload: SaveAllRequest) -> dict:
    snapshot = _require_dataset()
    saved_count = 0
    errors: List[dict] = []

    for item in payload.items:
        try:
            _ensure_image_known(snapshot, item.path)
            save_annotations(
                snapshot.dataset_dir,  # type: ignore[arg-type]
                item.path,
                item.segments,
                labels_dir=snapshot.labels_dir,
            )
            saved_count += 1
        except (HTTPException, FileNotFoundError, ValueError) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            errors.append({"path": item.path, "error": detail})

    return {"saved_count": saved_count, "error_count": len(errors), "errors": errors}


@app.post("/api/predict")
def predict(
    payload: PredictRequest,
    path: str = Query(..., description="Relative image path inside dataset"),
) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    image_path = image_absolute_path(snapshot.dataset_dir, path)  # type: ignore[arg-type]

    if payload.model_path:
        model_path = resolve_path(snapshot.dataset_dir, payload.model_path)  # type: ignore[arg-type]
    elif snapshot.model_path:
        model_path = snapshot.model_path
    else:
        raise HTTPException(status_code=400, detail="Model path is not set. Provide it in session or predict request.")

    sam2_config_path = None
    if payload.sam2_config:
        sam2_config_path = resolve_path(snapshot.dataset_dir, payload.sam2_config)  # type: ignore[arg-type]

    try:
        predicted_segments, model_classes = _runner.predict(
            image_path=image_path,
            model_path=model_path,
            conf=payload.conf,
            engine=payload.engine,
            model_type=payload.model_type,
            sam2_config=sam2_config_path,
            points_per_side=payload.points_per_side,
            points_per_batch=payload.points_per_batch,
            pred_iou_thresh=payload.pred_iou_thresh,
            stability_score_thresh=payload.stability_score_thresh,
            stability_score_offset=payload.stability_score_offset,
            box_nms_thresh=payload.box_nms_thresh,
            crop_n_layers=payload.crop_n_layers,
            crop_nms_thresh=payload.crop_nms_thresh,
            crop_overlap_ratio=payload.crop_overlap_ratio,
            crop_n_points_downscale_factor=payload.crop_n_points_downscale_factor,
            min_area=payload.min_area,
            simplify=payload.simplify,
            merge_same_class=payload.merge_same_class,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    class_list = list(snapshot.classes)
    if not class_list and model_classes:
        max_index = max(model_classes.keys())
        class_list = [model_classes.get(i, str(i)) for i in range(max_index + 1)]
        with _lock:
            _state.classes = class_list

    return {
        "segments": [segment.model_dump() for segment in predicted_segments],
        "model_classes": model_classes,
        "classes": class_list,
    }


@app.post("/api/predict_points")
def predict_points(
    payload: PointPredictRequest,
    path: str = Query(..., description="Relative image path inside dataset"),
) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    image_path = image_absolute_path(snapshot.dataset_dir, path)  # type: ignore[arg-type]

    if payload.model_path:
        model_path = resolve_path(snapshot.dataset_dir, payload.model_path)  # type: ignore[arg-type]
    elif snapshot.model_path:
        model_path = snapshot.model_path
    else:
        raise HTTPException(status_code=400, detail="Model path is not set. Provide it in session or request.")

    if not payload.sam2_config:
        raise HTTPException(status_code=400, detail="SAM 2 config YAML is required for point prompts.")
    sam2_config_path = resolve_path(snapshot.dataset_dir, payload.sam2_config)  # type: ignore[arg-type]

    points = [[float(item.x), float(item.y)] for item in payload.points]
    labels = [int(item.label) for item in payload.points]
    if not points:
        raise HTTPException(status_code=400, detail="At least one point is required.")
    if not any(label == 1 for label in labels):
        raise HTTPException(status_code=400, detail="At least one positive point is required.")

    try:
        segments = _runner.predict_sam2_points(
            image_path=image_path,
            model_path=model_path,
            sam2_config=sam2_config_path,
            points=points,
            labels=labels,
            multimask=payload.multimask,
            min_area=payload.min_area,
            simplify=payload.simplify,
            merge_contours=payload.merge_contours,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"segments": [segment.model_dump() for segment in segments]}


@app.delete("/api/annotations")
def clear_annotations(path: str = Query(..., description="Relative image path inside dataset")) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    empty: List[Segment] = []
    output_path = save_annotations(
        snapshot.dataset_dir,  # type: ignore[arg-type]
        path,
        empty,
        labels_dir=snapshot.labels_dir,
    )
    return {"saved": True, "label_path": str(output_path)}


@app.post("/api/video/session")
def set_video_session(payload: VideoSessionRequest) -> dict:
    video_path = resolve_path(Path.cwd(), payload.video_path)
    try:
        info = prepare_video_dataset(
            video_path=video_path,
            stride=payload.frame_stride,
            max_frames=payload.max_frames,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    classes = load_classes(info.dataset_dir, payload.classes, payload.classes_file)

    labels_dir = None
    if payload.labels_dir:
        labels_root = resolve_path(info.dataset_dir, payload.labels_dir)
        if labels_root.exists() and not labels_root.is_dir():
            raise HTTPException(status_code=400, detail=f"Labels path is not a directory: {labels_root}")
        labels_dir = str(labels_root)

    with _lock:
        _state.dataset_dir = info.dataset_dir
        _state.images = list(info.frames)
        _state.classes = classes
        _state.labels_dir = labels_dir
        _video_state.info = info

    _runner.reset_video()
    response = _session_payload(_state_snapshot())
    response["video"] = info.to_payload()
    return response


@app.post("/api/video/points")
def video_points(payload: VideoPointRequest) -> dict:
    info = _require_video()
    snapshot = _require_dataset()

    frame_idx = int(payload.frame_index)
    if frame_idx < 0 or frame_idx >= len(info.frames):
        raise HTTPException(status_code=400, detail=f"Frame index out of range: {frame_idx}")

    if payload.model_path:
        model_path = resolve_path(snapshot.dataset_dir, payload.model_path)  # type: ignore[arg-type]
    elif snapshot.model_path:
        model_path = snapshot.model_path
    else:
        raise HTTPException(status_code=400, detail="Model path is not set. Provide it in session or request.")

    if not payload.sam2_config:
        raise HTTPException(status_code=400, detail="SAM 2 config YAML is required for point prompts.")
    sam2_config_path = resolve_path(snapshot.dataset_dir, payload.sam2_config)  # type: ignore[arg-type]

    points = [[float(item.x), float(item.y)] for item in payload.points]
    labels = [int(item.label) for item in payload.points]
    if not points:
        raise HTTPException(status_code=400, detail="At least one point is required.")
    if not any(label == 1 for label in labels):
        raise HTTPException(status_code=400, detail="At least one positive point is required.")

    try:
        segments = _runner.predict_video_points(
            frames_dir=info.frames_dir,
            frame_idx=frame_idx,
            model_path=model_path,
            sam2_config=sam2_config_path,
            points=points,
            labels=labels,
            clear_old_points=payload.clear_old_points,
            min_area=payload.min_area,
            simplify=payload.simplify,
            merge_contours=payload.merge_contours,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"segments": [segment.model_dump() for segment in segments]}


@app.post("/api/video/propagate")
def video_propagate(payload: VideoPropagateRequest) -> dict:
    info = _require_video()
    snapshot = _require_dataset()

    if payload.model_path:
        model_path = resolve_path(snapshot.dataset_dir, payload.model_path)  # type: ignore[arg-type]
    elif snapshot.model_path:
        model_path = snapshot.model_path
    else:
        raise HTTPException(status_code=400, detail="Model path is not set. Provide it in session or request.")

    if not payload.sam2_config:
        raise HTTPException(status_code=400, detail="SAM 2 config YAML is required for video propagation.")
    sam2_config_path = resolve_path(snapshot.dataset_dir, payload.sam2_config)  # type: ignore[arg-type]

    if payload.start_frame is not None and (
        payload.start_frame < 0 or payload.start_frame >= len(info.frames)
    ):
        raise HTTPException(status_code=400, detail=f"Start frame out of range: {payload.start_frame}")

    seed_segments = payload.seed_segments or []
    seed_frame = payload.start_frame
    seed_frame_path = None
    if seed_segments:
        if seed_frame is None:
            seed_frame = 0
        if seed_frame < 0 or seed_frame >= len(info.frames):
            raise HTTPException(status_code=400, detail=f"Seed frame out of range: {seed_frame}")
        rel_path = info.frames[seed_frame]
        seed_frame_path = image_absolute_path(snapshot.dataset_dir, rel_path)  # type: ignore[arg-type]

    try:
        results = _runner.propagate_video(
            frames_dir=info.frames_dir,
            model_path=model_path,
            sam2_config=sam2_config_path,
            start_frame=seed_frame if seed_segments else payload.start_frame,
            max_frames=payload.max_frames,
            reverse=payload.reverse,
            min_area=payload.min_area,
            simplify=payload.simplify,
            merge_contours=payload.merge_contours,
            seed_segments=seed_segments,
            seed_frame_path=seed_frame_path,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    class_id = int(payload.class_id)
    saved_count = 0
    errors: List[dict] = []
    for frame_idx, segments in results:
        if segments:
            for segment in segments:
                segment.class_id = class_id
        if frame_idx < 0 or frame_idx >= len(info.frames):
            continue
        rel_path = info.frames[frame_idx]
        try:
            save_annotations(
                snapshot.dataset_dir,  # type: ignore[arg-type]
                rel_path,
                segments,
                labels_dir=snapshot.labels_dir,
            )
            saved_count += 1
        except (FileNotFoundError, ValueError) as exc:
            errors.append({"frame": frame_idx, "path": rel_path, "error": str(exc)})

    return {
        "processed": len(results),
        "saved_count": saved_count,
        "error_count": len(errors),
        "errors": errors,
    }


@app.get("/api/fs/roots")
def get_fs_roots() -> dict:
    return {"roots": _list_roots()}


@app.get("/api/fs/list")
def list_fs(
    path: str | None = Query(default=None, description="Directory path to inspect"),
    mode: str = Query(default="all", description="all|dir|model|yaml|video"),
) -> dict:
    if mode not in {"all", "dir", "model", "yaml", "video"}:
        raise HTTPException(status_code=400, detail="Unsupported mode. Use all, dir, model, yaml or video.")

    current = resolve_path(Path.cwd(), path) if path else Path.cwd().resolve()
    if not current.exists() or not current.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {current}")

    try:
        entries = list(current.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {current}") from exc

    directories = sorted([entry for entry in entries if entry.is_dir()], key=lambda item: item.name.lower())
    files = sorted([entry for entry in entries if entry.is_file()], key=lambda item: item.name.lower())

    if mode == "model":
        files = [entry for entry in files if entry.suffix.lower() in {".pt", ".onnx", ".pth", ".ckpt"}]
    elif mode == "yaml":
        files = [entry for entry in files if entry.suffix.lower() in {".yaml", ".yml", ".txt"}]
    elif mode == "video":
        files = [entry for entry in files if entry.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    elif mode == "dir":
        files = []

    parent = current.parent
    has_parent = parent != current

    return {
        "current_path": str(current),
        "parent_path": str(parent) if has_parent else None,
        "directories": [{"name": entry.name, "path": str(entry)} for entry in directories],
        "files": [{"name": entry.name, "path": str(entry)} for entry in files],
    }
