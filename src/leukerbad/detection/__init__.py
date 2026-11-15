"""Object detection components."""

from leukerbad.detection.base import Detection, Detector, GraspCandidate
from leukerbad.detection.geometry import (
    compute_grasp_from_polygon,
    estimate_gripper_opening_mm,
    normalize_axial_angle_deg,
)
from leukerbad.detection.tiled_detector import (
    TiledYoloSegDetector,
    apply_clahe_bgr,
    generate_tile_windows,
)
from leukerbad.detection.visualize import draw_detections
from leukerbad.detection.vlm_scene import (
    HybridVlmDetector,
    SceneDiscoveryCandidate,
    extract_contour_grasp_in_roi,
    rank_grasp_candidates,
)
from leukerbad.detection.vlm_verifier import VlmVerdict, VlmVerifiedDetector, crop_detection_roi
from leukerbad.detection.yolo_seg import YoloSegDetector, select_best_device

__all__ = [
    "Detection",
    "Detector",
    "GraspCandidate",
    "HybridVlmDetector",
    "SceneDiscoveryCandidate",
    "TiledYoloSegDetector",
    "VlmVerdict",
    "VlmVerifiedDetector",
    "YoloSegDetector",
    "apply_clahe_bgr",
    "compute_grasp_from_polygon",
    "crop_detection_roi",
    "draw_detections",
    "estimate_gripper_opening_mm",
    "extract_contour_grasp_in_roi",
    "generate_tile_windows",
    "normalize_axial_angle_deg",
    "rank_grasp_candidates",
    "select_best_device",
]
