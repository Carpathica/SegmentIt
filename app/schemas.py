from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Segment(BaseModel):
    class_id: int = Field(ge=0)
    points: List[List[float]] = Field(default_factory=list)
    score: Optional[float] = None


class SessionRequest(BaseModel):
    dataset_dir: str
    classes: List[str] = Field(default_factory=list)
    classes_file: Optional[str] = None
    model_path: Optional[str] = None
    labels_dir: Optional[str] = None


class SaveAnnotationsRequest(BaseModel):
    segments: List[Segment] = Field(default_factory=list)


class ImageAnnotations(BaseModel):
    path: str
    segments: List[Segment] = Field(default_factory=list)


class SaveAllRequest(BaseModel):
    items: List[ImageAnnotations] = Field(default_factory=list)


class PredictRequest(BaseModel):
    model_path: Optional[str] = None
    sam2_config: Optional[str] = None
    engine: str = Field(default="sam")
    conf: float = Field(default=0.25, ge=0.0, le=1.0)
    model_type: Optional[str] = None
    points_per_side: int = Field(default=32, ge=4, le=128)
    points_per_batch: int = Field(default=64, ge=1, le=1024)
    pred_iou_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    stability_score_thresh: float = Field(default=0.95, ge=0.0, le=1.0)
    stability_score_offset: float = Field(default=1.0, ge=0.0, le=3.0)
    box_nms_thresh: float = Field(default=0.7, ge=0.0, le=1.0)
    crop_n_layers: int = Field(default=0, ge=0, le=6)
    crop_nms_thresh: float = Field(default=0.7, ge=0.0, le=1.0)
    crop_overlap_ratio: float = Field(default=512 / 1500, ge=0.0, le=1.0)
    crop_n_points_downscale_factor: int = Field(default=1, ge=1, le=8)
    min_area: int = Field(default=200, ge=0)
    simplify: float = Field(default=0.01, ge=0.0, le=0.1)
    merge_same_class: bool = Field(default=False)


class PointPrompt(BaseModel):
    x: float
    y: float
    label: int = Field(ge=0, le=1)


class PointPredictRequest(BaseModel):
    model_path: Optional[str] = None
    sam2_config: Optional[str] = None
    points: List[PointPrompt] = Field(default_factory=list)
    multimask: bool = True
    min_area: int = Field(default=0, ge=0)
    simplify: float = Field(default=0.01, ge=0.0, le=0.1)
    merge_contours: bool = Field(default=True)


class VideoSessionRequest(BaseModel):
    video_path: str
    frame_stride: int = Field(default=1, ge=1, le=60)
    max_frames: Optional[int] = Field(default=None, ge=1)
    labels_dir: Optional[str] = None
    classes: List[str] = Field(default_factory=list)
    classes_file: Optional[str] = None


class VideoPointRequest(BaseModel):
    model_path: Optional[str] = None
    sam2_config: Optional[str] = None
    frame_index: int = Field(ge=0)
    points: List[PointPrompt] = Field(default_factory=list)
    clear_old_points: bool = True
    min_area: int = Field(default=0, ge=0)
    simplify: float = Field(default=0.01, ge=0.0, le=0.1)
    merge_contours: bool = Field(default=True)


class VideoPropagateRequest(BaseModel):
    model_path: Optional[str] = None
    sam2_config: Optional[str] = None
    start_frame: Optional[int] = Field(default=None, ge=0)
    max_frames: Optional[int] = Field(default=None, ge=1)
    reverse: bool = False
    min_area: int = Field(default=0, ge=0)
    simplify: float = Field(default=0.01, ge=0.0, le=0.1)
    class_id: int = Field(default=0, ge=0)
    merge_contours: bool = Field(default=True)
    seed_segments: List[Segment] = Field(default_factory=list)
