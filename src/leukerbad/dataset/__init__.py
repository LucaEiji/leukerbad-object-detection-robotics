"""Dataset A capture, SAM auto-masking, and dataset integrity validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leukerbad.dataset.capture import VALID_SPLITS, build_capture_filename, save_labeled_frame
    from leukerbad.dataset.quality_filter import (
        ImageQualityMetrics,
        QualityFilterReport,
        assess_image_quality,
        compute_polygon_elongation,
        filter_dataset_split_by_quality,
        find_underrepresented_shape_stems,
    )
    from leukerbad.dataset.sam_masker import (
        ParsedYoloLine,
        parse_yolo_label_line,
        polygon_px_to_yolo_seg_line,
        upgrade_bbox_labels_with_sam,
        yolo_bbox_to_xyxy_px,
    )
    from leukerbad.dataset.validate import (
        ValidationReport,
        validate_yolo_seg_dataset,
        write_yolo_dataset_yaml,
    )

__all__ = [
    "VALID_SPLITS",
    "ImageQualityMetrics",
    "ParsedYoloLine",
    "QualityFilterReport",
    "ValidationReport",
    "assess_image_quality",
    "build_capture_filename",
    "compute_polygon_elongation",
    "filter_dataset_split_by_quality",
    "find_underrepresented_shape_stems",
    "parse_yolo_label_line",
    "polygon_px_to_yolo_seg_line",
    "save_labeled_frame",
    "upgrade_bbox_labels_with_sam",
    "validate_yolo_seg_dataset",
    "write_yolo_dataset_yaml",
    "yolo_bbox_to_xyxy_px",
]


def __getattr__(name: str) -> Any:
    if name in ("VALID_SPLITS", "build_capture_filename", "save_labeled_frame"):
        from leukerbad.dataset import capture

        return getattr(capture, name)
    if name in (
        "ImageQualityMetrics",
        "QualityFilterReport",
        "assess_image_quality",
        "compute_polygon_elongation",
        "filter_dataset_split_by_quality",
        "find_underrepresented_shape_stems",
    ):
        from leukerbad.dataset import quality_filter

        return getattr(quality_filter, name)
    if name in (
        "ParsedYoloLine",
        "parse_yolo_label_line",
        "polygon_px_to_yolo_seg_line",
        "upgrade_bbox_labels_with_sam",
        "yolo_bbox_to_xyxy_px",
    ):
        from leukerbad.dataset import sam_masker

        return getattr(sam_masker, name)
    if name in ("ValidationReport", "validate_yolo_seg_dataset", "write_yolo_dataset_yaml"):
        from leukerbad.dataset import validate

        return getattr(validate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
