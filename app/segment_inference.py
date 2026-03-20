from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .schemas import Segment

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency import errors
    cv2 = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - optional dependency import errors
    torch = None  # type: ignore[assignment]

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except Exception:  # pragma: no cover - optional dependency import errors
    SamAutomaticMaskGenerator = None  # type: ignore[assignment]
    sam_model_registry = None  # type: ignore[assignment]

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_hf, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:  # pragma: no cover - optional dependency import errors
    SAM2AutomaticMaskGenerator = None  # type: ignore[assignment]
    build_sam2 = None  # type: ignore[assignment]
    build_sam2_hf = None  # type: ignore[assignment]
    build_sam2_video_predictor = None  # type: ignore[assignment]
    SAM2ImagePredictor = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency import errors
    YOLO = None  # type: ignore[assignment]


def _filter_kwargs(target, kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(kwargs)
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def _segments_to_mask(segments: List[Segment], width: int, height: int) -> np.ndarray | None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for mask rasterization.")
    if not segments:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for segment in segments:
        if not segment.points or len(segment.points) < 3:
            continue
        points = np.array(segment.points, dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygon = points.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [polygon], 1)
    if np.any(mask):
        return mask.astype(bool)
    return None


def _infer_sam_model_type(model_path: Path) -> str | None:
    name = model_path.name.lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"
    return None


def _mask_to_segments(
    mask: np.ndarray,
    class_id: int,
    score: float | None,
    simplify: float,
    min_area: int = 0,
    merge_contours: bool = False,
) -> List[Segment]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for SAM mask conversion.")

    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = np.any(mask, axis=0)

    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: List[Segment] = []
    filtered: List[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        if min_area and cv2.contourArea(contour) < float(min_area):
            continue
        filtered.append(contour)

    if not filtered:
        return segments

    if merge_contours and len(filtered) > 1:
        all_points = np.vstack(filtered)
        hull = cv2.convexHull(all_points)
        approx = hull
        if simplify > 0:
            epsilon = float(simplify) * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
        points = approx.reshape(-1, 2).tolist()
        if len(points) >= 3:
            segments.append(
                Segment(
                    class_id=class_id,
                    points=[[float(x), float(y)] for x, y in points],
                    score=score,
                )
            )
        return segments

    for contour in filtered:
        approx = contour
        if simplify > 0:
            epsilon = float(simplify) * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2).tolist()
        if len(points) < 3:
            continue
        segments.append(
            Segment(
                class_id=class_id,
                points=[[float(x), float(y)] for x, y in points],
                score=score,
            )
        )
    return segments


class SegmentRunner:
    def __init__(self) -> None:
        self._sam = None
        self._sam_model_path: Path | None = None
        self._sam_model_type: str | None = None
        self._sam_device: str | None = None
        self._sam_generator = None
        self._sam_params: Dict[str, object] | None = None

        self._sam2 = None
        self._sam2_model_path: Path | None = None
        self._sam2_config: str | None = None
        self._sam2_device: str | None = None
        self._sam2_generator = None
        self._sam2_params: Dict[str, object] | None = None
        self._sam2_point_predictor = None
        self._sam2_point_model_path: Path | None = None
        self._sam2_point_config: str | None = None
        self._sam2_point_device: str | None = None
        self._sam2_point_image_path: Path | None = None

        self._sam2_video_predictor = None
        self._sam2_video_model_path: Path | None = None
        self._sam2_video_config: str | None = None
        self._sam2_video_device: str | None = None
        self._sam2_video_state = None
        self._sam2_video_frames_dir: Path | None = None

        self._yolo = None
        self._yolo_model_path: Path | None = None

    def predict(
        self,
        image_path: Path,
        model_path: Path,
        engine: str = "sam",
        conf: float = 0.25,
        model_type: str | None = None,
        sam2_config: Path | None = None,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        min_area: int = 200,
        simplify: float = 0.01,
        merge_same_class: bool = False,
    ) -> Tuple[List[Segment], Dict[int, str]]:
        engine_name = (engine or "sam").lower()
        if engine_name in {"sam", "segment-anything", "segment_anything"}:
            return self._predict_sam(
                image_path=image_path,
                model_path=model_path,
                model_type=model_type,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crop_n_layers,
                crop_nms_thresh=crop_nms_thresh,
                crop_overlap_ratio=crop_overlap_ratio,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_area=min_area,
                simplify=simplify,
                merge_same_class=merge_same_class,
            )
        if engine_name in {"sam2", "sam-2", "segment-anything-2", "segment_anything_2"}:
            return self._predict_sam2(
                image_path=image_path,
                model_path=model_path,
                sam2_config=sam2_config,
                points_per_side=points_per_side,
                points_per_batch=points_per_batch,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                box_nms_thresh=box_nms_thresh,
                crop_n_layers=crop_n_layers,
                crop_nms_thresh=crop_nms_thresh,
                crop_overlap_ratio=crop_overlap_ratio,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_area=min_area,
                simplify=simplify,
                merge_same_class=merge_same_class,
            )
        if engine_name in {"yolo", "yolo-seg", "ultralytics"}:
            return self._predict_yolo(
                image_path=image_path,
                model_path=model_path,
                conf=conf,
                min_area=min_area,
                simplify=simplify,
                merge_same_class=merge_same_class,
            )
        raise ValueError(f"Unsupported segmentation engine: {engine}")

    def _predict_sam(
        self,
        image_path: Path,
        model_path: Path,
        model_type: str | None,
        points_per_side: int,
        points_per_batch: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        stability_score_offset: float,
        box_nms_thresh: float,
        crop_n_layers: int,
        crop_nms_thresh: float,
        crop_overlap_ratio: float,
        crop_n_points_downscale_factor: int,
        min_area: int,
        simplify: float,
        merge_same_class: bool,
    ) -> Tuple[List[Segment], Dict[int, str]]:
        if SamAutomaticMaskGenerator is None or sam_model_registry is None or torch is None:
            raise RuntimeError(
                "segment-anything is not installed. Install dependencies from sam_web_annotator/requirements.txt"
            )
        if cv2 is None:
            raise RuntimeError("opencv-python is required for SAM mask conversion.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        resolved_model_type = model_type or _infer_sam_model_type(model_path) or "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        generator_params = _filter_kwargs(
            SamAutomaticMaskGenerator,
            {
                "points_per_side": int(points_per_side),
                "points_per_batch": int(points_per_batch),
                "pred_iou_thresh": float(pred_iou_thresh),
                "stability_score_thresh": float(stability_score_thresh),
                "stability_score_offset": float(stability_score_offset),
                "box_nms_thresh": float(box_nms_thresh),
                "crop_n_layers": int(crop_n_layers),
                "crop_nms_thresh": float(crop_nms_thresh),
                "crop_overlap_ratio": float(crop_overlap_ratio),
                "crop_n_points_downscale_factor": int(crop_n_points_downscale_factor),
                "min_mask_region_area": int(min_area),
            },
        )

        if (
            self._sam is None
            or self._sam_model_path != model_path
            or self._sam_model_type != resolved_model_type
            or self._sam_device != device
            or self._sam_params != generator_params
        ):
            sam = sam_model_registry[resolved_model_type](checkpoint=str(model_path))
            sam.to(device)
            self._sam = sam
            self._sam_model_path = model_path
            self._sam_model_type = resolved_model_type
            self._sam_device = device
            self._sam_generator = SamAutomaticMaskGenerator(sam, **generator_params)
            self._sam_params = generator_params

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self._sam_generator.generate(image_rgb)
        segments: List[Segment] = []
        merged_masks: List[np.ndarray] = []
        merged_score: float | None = None
        for item in masks:
            mask = item.get("segmentation")
            if mask is None:
                continue
            area = int(item.get("area") or int(np.sum(mask)))
            if min_area and area < int(min_area):
                continue
            score = item.get("predicted_iou")
            if score is None:
                score = item.get("stability_score")
            if merge_same_class:
                merged_masks.append(mask)
                if score is not None:
                    merged_score = score if merged_score is None else max(float(score), merged_score)
            else:
                segments.extend(_mask_to_segments(mask, class_id=0, score=score, simplify=simplify, min_area=min_area))
        if merge_same_class:
            if not merged_masks:
                return [], {}
            merged = np.zeros_like(merged_masks[0], dtype=bool)
            for mask in merged_masks:
                merged |= mask.astype(bool)
            segments = _mask_to_segments(
                merged,
                class_id=0,
                score=merged_score,
                simplify=simplify,
                min_area=min_area,
                merge_contours=True,
            )
        return segments, {}

    def _predict_sam2(
        self,
        image_path: Path,
        model_path: Path,
        sam2_config: Path | None,
        points_per_side: int,
        points_per_batch: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        stability_score_offset: float,
        box_nms_thresh: float,
        crop_n_layers: int,
        crop_nms_thresh: float,
        crop_overlap_ratio: float,
        crop_n_points_downscale_factor: int,
        min_area: int,
        simplify: float,
        merge_same_class: bool,
    ) -> Tuple[List[Segment], Dict[int, str]]:
        if SAM2AutomaticMaskGenerator is None or build_sam2 is None or torch is None:
            raise RuntimeError(
                "sam2 is not installed. Install dependencies from https://github.com/facebookresearch/sam2"
            )
        if cv2 is None:
            raise RuntimeError("opencv-python is required for SAM 2 mask conversion.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if sam2_config is None:
            raise ValueError("SAM 2 config YAML is required (e.g. configs/sam2.1/sam2.1_hiera_l.yaml).")

        config_value = str(sam2_config)
        if sam2_config.is_absolute() and not sam2_config.exists():
            raise FileNotFoundError(f"SAM 2 config not found: {sam2_config}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        generator_params = _filter_kwargs(
            SAM2AutomaticMaskGenerator,
            {
                "points_per_side": int(points_per_side),
                "points_per_batch": int(points_per_batch),
                "pred_iou_thresh": float(pred_iou_thresh),
                "stability_score_thresh": float(stability_score_thresh),
                "stability_score_offset": float(stability_score_offset),
                "box_nms_thresh": float(box_nms_thresh),
                "crop_n_layers": int(crop_n_layers),
                "crop_nms_thresh": float(crop_nms_thresh),
                "crop_overlap_ratio": float(crop_overlap_ratio),
                "crop_n_points_downscale_factor": int(crop_n_points_downscale_factor),
                "min_mask_region_area": int(min_area),
            },
        )

        if (
            self._sam2 is None
            or self._sam2_model_path != model_path
            or self._sam2_config != config_value
            or self._sam2_device != device
            or self._sam2_params != generator_params
        ):
            sam2 = build_sam2(config_value, ckpt_path=str(model_path), device=device)
            self._sam2 = sam2
            self._sam2_model_path = model_path
            self._sam2_config = config_value
            self._sam2_device = device
            self._sam2_generator = SAM2AutomaticMaskGenerator(sam2, **generator_params)
            self._sam2_params = generator_params

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self._sam2_generator.generate(image_rgb)
        segments: List[Segment] = []
        merged_masks: List[np.ndarray] = []
        merged_score: float | None = None
        for item in masks:
            mask = item.get("segmentation")
            if mask is None:
                continue
            area = int(item.get("area") or int(np.sum(mask)))
            if min_area and area < int(min_area):
                continue
            score = item.get("predicted_iou")
            if score is None:
                score = item.get("stability_score")
            if merge_same_class:
                merged_masks.append(mask)
                if score is not None:
                    merged_score = score if merged_score is None else max(float(score), merged_score)
            else:
                segments.extend(_mask_to_segments(mask, class_id=0, score=score, simplify=simplify, min_area=min_area))
        if merge_same_class:
            if not merged_masks:
                return [], {}
            merged = np.zeros_like(merged_masks[0], dtype=bool)
            for mask in merged_masks:
                merged |= mask.astype(bool)
            segments = _mask_to_segments(
                merged,
                class_id=0,
                score=merged_score,
                simplify=simplify,
                min_area=min_area,
                merge_contours=True,
            )
        return segments, {}

    def reset_video(self) -> None:
        self._sam2_video_state = None
        self._sam2_video_frames_dir = None
        self._sam2_video_predictor = None
        self._sam2_video_model_path = None
        self._sam2_video_config = None
        self._sam2_video_device = None

    def _ensure_sam2_point_predictor(
        self,
        model_path: Path,
        sam2_config: Path,
        device: str,
    ) -> None:
        if SAM2ImagePredictor is None or build_sam2 is None or torch is None:
            raise RuntimeError(
                "sam2 is not installed. Install dependencies from https://github.com/facebookresearch/sam2"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        config_value = str(sam2_config)
        if sam2_config.is_absolute() and not sam2_config.exists():
            raise FileNotFoundError(f"SAM 2 config not found: {sam2_config}")

        if (
            self._sam2_point_predictor is None
            or self._sam2_point_model_path != model_path
            or self._sam2_point_config != config_value
            or self._sam2_point_device != device
        ):
            sam2 = build_sam2(config_value, ckpt_path=str(model_path), device=device)
            self._sam2_point_predictor = SAM2ImagePredictor(sam2)
            self._sam2_point_model_path = model_path
            self._sam2_point_config = config_value
            self._sam2_point_device = device
            self._sam2_point_image_path = None

    def predict_sam2_points(
        self,
        image_path: Path,
        model_path: Path,
        sam2_config: Path,
        points: List[List[float]],
        labels: List[int],
        multimask: bool = True,
        min_area: int = 0,
        simplify: float = 0.01,
        merge_contours: bool = True,
    ) -> List[Segment]:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for SAM 2 mask conversion.")
        if torch is None:
            raise RuntimeError("torch is not available.")
        if not points:
            raise ValueError("At least one point is required for point segmentation.")
        if len(points) != len(labels):
            raise ValueError("Points and labels must have the same length.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ensure_sam2_point_predictor(model_path, sam2_config, device)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._sam2_point_image_path != image_path:
            self._sam2_point_predictor.set_image(image_rgb)
            self._sam2_point_image_path = image_path

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)
        masks, ious, _ = self._sam2_point_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
            return_logits=False,
            normalize_coords=True,
        )
        if masks is None or len(masks) == 0:
            return []

        if masks.ndim == 2:
            masks = masks[None, ...]
        if ious is None or len(ious) == 0:
            best_index = 0
            score = None
        else:
            best_index = int(np.argmax(ious))
            score = float(ious[best_index])

        best_mask = masks[best_index]
        mask_area = int(np.sum(best_mask))
        if min_area and mask_area < int(min_area):
            return []
        segments = _mask_to_segments(
            best_mask,
            class_id=0,
            score=score,
            simplify=simplify,
            min_area=min_area,
            merge_contours=merge_contours,
        )
        return segments

    def _ensure_sam2_video_predictor(
        self,
        frames_dir: Path,
        model_path: Path,
        sam2_config: Path,
        device: str,
    ) -> None:
        if build_sam2_video_predictor is None or torch is None:
            raise RuntimeError(
                "sam2 is not installed. Install dependencies from https://github.com/facebookresearch/sam2"
            )
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        config_value = str(sam2_config)
        if sam2_config.is_absolute() and not sam2_config.exists():
            raise FileNotFoundError(f"SAM 2 config not found: {sam2_config}")

        if (
            self._sam2_video_predictor is None
            or self._sam2_video_model_path != model_path
            or self._sam2_video_config != config_value
            or self._sam2_video_device != device
        ):
            self._sam2_video_predictor = build_sam2_video_predictor(
                config_file=config_value,
                ckpt_path=str(model_path),
                device=device,
            )
            self._sam2_video_model_path = model_path
            self._sam2_video_config = config_value
            self._sam2_video_device = device
            self._sam2_video_state = None

        if self._sam2_video_state is None or self._sam2_video_frames_dir != frames_dir:
            self._sam2_video_state = self._sam2_video_predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=True,
                async_loading_frames=True,
            )
            self._sam2_video_frames_dir = frames_dir

    def predict_video_points(
        self,
        frames_dir: Path,
        frame_idx: int,
        model_path: Path,
        sam2_config: Path,
        points: List[List[float]],
        labels: List[int],
        clear_old_points: bool = True,
        min_area: int = 0,
        simplify: float = 0.01,
        merge_contours: bool = True,
    ) -> List[Segment]:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for SAM 2 mask conversion.")
        if torch is None:
            raise RuntimeError("torch is not available.")
        if not points:
            raise ValueError("At least one point is required for point segmentation.")
        if len(points) != len(labels):
            raise ValueError("Points and labels must have the same length.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ensure_sam2_video_predictor(frames_dir, model_path, sam2_config, device)
        inference_state = self._sam2_video_state
        if inference_state is None:
            raise RuntimeError("Video inference state is not initialized.")

        _, obj_ids, video_res_masks = self._sam2_video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(frame_idx),
            obj_id=0,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points,
            normalize_coords=True,
        )
        if not obj_ids or video_res_masks is None:
            return []

        mask_tensor = video_res_masks[0]
        mask_np = mask_tensor.detach().float().cpu().numpy() > 0
        mask_area = int(np.sum(mask_np))
        if min_area and mask_area < int(min_area):
            return []
        return _mask_to_segments(
            mask_np,
            class_id=0,
            score=None,
            simplify=simplify,
            min_area=min_area,
            merge_contours=merge_contours,
        )

    def propagate_video(
        self,
        frames_dir: Path,
        model_path: Path,
        sam2_config: Path,
        start_frame: int | None,
        max_frames: int | None,
        reverse: bool,
        min_area: int = 0,
        simplify: float = 0.01,
        merge_contours: bool = True,
        seed_segments: List[Segment] | None = None,
        seed_frame_path: Path | None = None,
    ) -> List[Tuple[int, List[Segment]]]:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for SAM 2 mask conversion.")
        if torch is None:
            raise RuntimeError("torch is not available.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ensure_sam2_video_predictor(frames_dir, model_path, sam2_config, device)
        inference_state = self._sam2_video_state
        if inference_state is None:
            raise RuntimeError("Video inference state is not initialized.")

        if seed_segments and seed_frame_path is not None and start_frame is not None:
            frame = cv2.imread(str(seed_frame_path))
            if frame is None:
                raise ValueError(f"Failed to read frame: {seed_frame_path}")
            height, width = frame.shape[:2]
            seed_mask = _segments_to_mask(seed_segments, width=width, height=height)
            if seed_mask is not None:
                self._sam2_video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=int(start_frame),
                    obj_id=0,
                    mask=seed_mask,
                )

        results: List[Tuple[int, List[Segment]]] = []
        for frame_idx, _, video_res_masks in self._sam2_video_predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        ):
            mask_tensor = video_res_masks[0]
            mask_np = mask_tensor.detach().float().cpu().numpy() > 0
            mask_area = int(np.sum(mask_np))
            if min_area and mask_area < int(min_area):
                results.append((int(frame_idx), []))
                continue
            segments = _mask_to_segments(
                mask_np,
                class_id=0,
                score=None,
                simplify=simplify,
                min_area=min_area,
                merge_contours=merge_contours,
            )
            results.append((int(frame_idx), segments))
        return results

    def _predict_yolo(
        self,
        image_path: Path,
        model_path: Path,
        conf: float,
        min_area: int,
        simplify: float,
        merge_same_class: bool,
    ) -> Tuple[List[Segment], Dict[int, str]]:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Install dependencies from sam_web_annotator/requirements.txt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if self._yolo is None or self._yolo_model_path != model_path:
            self._yolo = YOLO(str(model_path))
            self._yolo_model_path = model_path

        results = self._yolo.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
        )
        if not results:
            return [], {}

        result = results[0]
        names = getattr(result, "names", {}) or {}
        class_names = {int(key): str(value) for key, value in names.items()}

        segments: List[Segment] = []
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        mask_polys = masks.xy if masks is not None else []
        for idx, polygon in enumerate(mask_polys):
            if polygon is None or len(polygon) < 3:
                continue
            class_id = 0
            score = None
            if boxes is not None and getattr(boxes, "cls", None) is not None and idx < len(boxes.cls):
                class_id = int(boxes.cls[idx].item())
            if boxes is not None and getattr(boxes, "conf", None) is not None and idx < len(boxes.conf):
                score = float(boxes.conf[idx].item())
            points = [[float(x), float(y)] for x, y in np.asarray(polygon).tolist()]
            segments.append(Segment(class_id=max(class_id, 0), points=points, score=score))
        if not merge_same_class or not segments:
            return segments, class_names
        if cv2 is None:
            return segments, class_names

        image = cv2.imread(str(image_path))
        if image is None:
            return segments, class_names
        height, width = image.shape[:2]
        merged_by_class: Dict[int, np.ndarray] = {}
        merged_score: Dict[int, float] = {}

        for segment in segments:
            class_id = int(segment.class_id)
            mask = merged_by_class.get(class_id)
            if mask is None:
                mask = np.zeros((height, width), dtype=np.uint8)
                merged_by_class[class_id] = mask
            polygon = np.array(segment.points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [polygon], 1)
            if segment.score is not None:
                merged_score[class_id] = max(segment.score, merged_score.get(class_id, segment.score))

        merged_segments: List[Segment] = []
        for class_id, mask in merged_by_class.items():
            merged_segments.extend(
                _mask_to_segments(
                    mask.astype(bool),
                    class_id=class_id,
                    score=merged_score.get(class_id),
                    simplify=simplify,
                    min_area=min_area,
                    merge_contours=True,
                )
            )
        return merged_segments, class_names
