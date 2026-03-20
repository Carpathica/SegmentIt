from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import List, Optional

import json

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency import errors
    cv2 = None  # type: ignore[assignment]


@dataclass
class VideoInfo:
    dataset_dir: Path
    frames_dir: Path
    labels_dir: Path
    frames: List[str]
    fps: float
    width: int
    height: int
    stride: int
    total_frames: int
    video_path: Path

    def to_payload(self) -> dict:
        return {
            "dataset_dir": str(self.dataset_dir),
            "frames_dir": str(self.frames_dir),
            "labels_dir": str(self.labels_dir),
            "frame_count": len(self.frames),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "stride": self.stride,
            "total_frames": self.total_frames,
            "video_path": str(self.video_path),
        }


def _cache_root() -> Path:
    return Path(__file__).resolve().parents[1] / ".cache" / "video_frames"


def _cache_key(video_path: Path, stride: int, max_frames: Optional[int]) -> str:
    stat = video_path.stat()
    raw = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime}|{stride}|{max_frames}"
    return md5(raw.encode("utf-8")).hexdigest()[:12]


def _write_meta(meta_path: Path, payload: dict) -> None:
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_meta(meta_path: Path) -> Optional[dict]:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def prepare_video_dataset(
    video_path: Path,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> VideoInfo:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video support.")
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cache_root = _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _cache_key(video_path, stride, max_frames)
    dataset_dir = cache_root / f"{video_path.stem}_{key}"
    frames_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    meta_path = dataset_dir / "video_meta.json"

    cached = _read_meta(meta_path)
    if cached and frames_dir.exists():
        frames = cached.get("frames", [])
        if frames:
            return VideoInfo(
                dataset_dir=dataset_dir,
                frames_dir=frames_dir,
                labels_dir=labels_dir,
                frames=list(frames),
                fps=float(cached.get("fps") or 0.0),
                width=int(cached.get("width") or 0),
                height=int(cached.get("height") or 0),
                stride=int(cached.get("stride") or stride),
                total_frames=int(cached.get("total_frames") or 0),
                video_path=video_path,
            )

    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[str] = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            name = f"{saved_idx:06d}.jpg"
            out_path = frames_dir / name
            cv2.imwrite(str(out_path), frame)
            frames.append(f"images/{name}")
            saved_idx += 1
            if max_frames and saved_idx >= max_frames:
                break
        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted from video.")

    meta_payload = {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "stride": stride,
        "total_frames": total_frames or frame_idx,
    }
    _write_meta(meta_path, meta_payload)

    return VideoInfo(
        dataset_dir=dataset_dir,
        frames_dir=frames_dir,
        labels_dir=labels_dir,
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        stride=stride,
        total_frames=total_frames or frame_idx,
        video_path=video_path,
    )
