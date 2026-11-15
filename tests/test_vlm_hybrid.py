"""Offline unit tests for batched VLM verification, scene discovery, contour grasp, and ranking."""

import importlib.machinery
import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pytest

from leukerbad.detection.base import Detection, GraspCandidate
from leukerbad.detection.geometry import compute_grasp_from_polygon
from leukerbad.detection.vlm_scene import (
    HybridVlmDetector,
    SceneDiscoveryCandidate,
    extract_contour_grasp_in_roi,
    rank_grasp_candidates,
)
from leukerbad.detection.vlm_verifier import (
    _SYSTEM_INSTRUCTION,
    VlmVerifiedDetector,
    _parse_batch_verdicts_json,
)


@dataclass
class _StubDetector:
    detections: list[Detection]
    mm_per_px: float | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        return list(self.detections)


@dataclass
class _FakeResponse:
    text: str


class _FakeModels:
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self,
        *,
        model: str,
        contents: list[Any],
        config: Any,
    ) -> _FakeResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(text=item)


class _FakeGenAiClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.models = _FakeModels(responses)


def _ensure_google_genai_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register stub `google.genai` module specs when running without `google-genai` installed."""
    try:
        has_real = importlib.util.find_spec("google.genai") is not None
    except (ModuleNotFoundError, ValueError):
        has_real = False
    if has_real:
        return

    google_mod = sys.modules.get("google")
    if google_mod is None:
        google_mod = types.ModuleType("google")
        google_mod.__path__ = []  # type: ignore[attr-defined]
        google_mod.__spec__ = importlib.machinery.ModuleSpec(
            "google",
            loader=None,
            is_package=True,
        )
        monkeypatch.setitem(sys.modules, "google", google_mod)

    genai_mod = types.ModuleType("google.genai")
    genai_mod.__spec__ = importlib.machinery.ModuleSpec("google.genai", loader=None)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setattr(google_mod, "genai", genai_mod, raising=False)


def _draw_rotated_puff_bar(
    frame: np.ndarray,
    center: tuple[float, float],
    length_px: float,
    width_px: float,
    angle_deg: float,
    color_bgr: tuple[int, int, int] = (215, 95, 25),
) -> np.ndarray:
    """Render a synthetic rotated puff bar onto `frame` and return its 4-corner polygon."""
    rect = (center, (length_px, width_px), angle_deg)
    box = cv2.boxPoints(rect)
    pts_i32 = np.round(box).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(frame, [pts_i32], color_bgr, lineType=cv2.LINE_AA)
    return box.astype(np.float32)


def test_upgraded_system_instruction_covers_diagnostic_lookalikes_and_box_puffs() -> None:
    lowered = _SYSTEM_INSTRUCTION.lower()
    assert "aukey" in lowered
    assert "elfbar" in lowered
    assert "cardboard" in lowered
    assert "bic" in lowered
    assert "duracell" in lowered
    assert "flashlight" in lowered
    assert "inhaler" in lowered
    assert "box-mod" in lowered


def test_batched_vlm_verifier_rescues_box_puff_and_rejects_packaging_and_pen() -> None:
    frame = np.full((360, 480, 3), 118, dtype=np.uint8)
    shiny_puff = Detection(
        label="vape",
        confidence=0.22,
        bounding_box=(30, 40, 110, 115),
    )
    aukey_box_fp = Detection(
        label="vape",
        confidence=0.74,
        bounding_box=(150, 50, 290, 130),
    )
    bic_pen_fp = Detection(
        label="vape",
        confidence=0.41,
        bounding_box=(320, 60, 440, 95),
    )

    batch_payload = json.dumps(
        [
            {
                "index": 0,
                "is_vape": True,
                "detected_object": "box_mod_puff",
                "confidence": 0.93,
                "hazard_note": "Embedded Li-ion cell + e-liquid fire hazard",
            },
            {
                "index": 1,
                "is_vape": False,
                "detected_object": "printed_packaging_graphic",
                "confidence": 0.97,
                "hazard_note": "2D webcam graphic printed on flat AUKEY cardboard box",
            },
            {
                "index": 2,
                "is_vape": False,
                "detected_object": "ballpoint_pen",
                "confidence": 0.95,
                "hazard_note": "Multi-color Bic click pen with clip",
            },
        ]
    )

    client = _FakeGenAiClient([batch_payload])
    verifier = VlmVerifiedDetector(
        base_detector=_StubDetector([shiny_puff, aukey_box_fp, bic_pen_fp]),
        min_conf_to_verify=0.15,
        max_conf_to_verify=0.88,
        batch_verify=True,
        reject_non_vapes=False,
        client=client,
    )
    results = verifier.detect(frame)

    # Single batched Gemini call containing 1 full-scene image + 3 ROI crops
    assert len(client.models.calls) == 1
    call_contents = client.models.calls[0]["contents"]
    image_parts = [part for part in call_contents if not isinstance(part, str)]
    assert len(image_parts) == 4

    assert len(results) == 3
    assert results[0].label == "vape"
    assert results[0].confidence == pytest.approx(round(0.4 * 0.22 + 0.6 * 0.93, 4))
    assert results[0].metadata["vlm"]["detected_object"] == "box_mod_puff"

    assert results[1].label == "non_vape:printed_packaging_graphic"
    assert results[1].confidence == pytest.approx(0.97)
    assert results[1].metadata["vlm"]["is_vape"] is False

    assert results[2].label == "non_vape:ballpoint_pen"
    assert results[2].confidence == pytest.approx(0.95)
    assert results[2].metadata["vlm"]["is_vape"] is False


def test_parse_batch_verdicts_handles_markdown_fences_and_missing_indices() -> None:
    fenced = """```json
    [
      {"index": 2, "is_vape": false, "detected_object": "aa_battery", "confidence": 0.91},
      {"index": 0, "is_vape": true, "detected_object": "upright_vape", "confidence": 0.88}
    ]
    ```"""
    parsed = _parse_batch_verdicts_json(fenced, expected_count=3)
    assert len(parsed) == 3
    assert parsed[0] is not None and parsed[0].is_vape is True
    assert parsed[1] is None
    assert parsed[2] is not None and parsed[2].detected_object == "aa_battery"


def test_extract_contour_grasp_in_roi_recovers_rotated_puff_orientation() -> None:
    frame = np.full((320, 420, 3), (124, 126, 128), dtype=np.uint8)
    target_angle_deg = -35.0
    box_pts = _draw_rotated_puff_bar(
        frame,
        center=(210.0, 160.0),
        length_px=140.0,
        width_px=36.0,
        angle_deg=target_angle_deg,
        color_bgr=(220, 90, 20),
    )

    # Loose axis-aligned VLM bounding box with generous tabletop margin
    x1 = int(np.floor(box_pts[:, 0].min())) - 16
    y1 = int(np.floor(box_pts[:, 1].min())) - 16
    x2 = int(np.ceil(box_pts[:, 0].max())) + 16
    y2 = int(np.ceil(box_pts[:, 1].max())) + 16

    polygon, grasp, meta = extract_contour_grasp_in_roi(
        frame,
        (x1, y1, x2, y2),
        mm_per_px=0.5,
    )

    assert meta["contour_method"] == "lab_foreground"
    assert polygon.shape[0] >= 4
    assert grasp is not None
    assert abs(grasp.object_angle_deg - target_angle_deg) <= 6.0
    assert abs(grasp.gripper_angle_deg - 55.0) <= 6.0
    assert 28.0 <= grasp.width_px <= 44.0
    assert "width_mm" in meta and "gripper_opening_mm" in meta


def test_extract_contour_grasp_falls_back_cleanly_on_uniform_or_tiny_roi() -> None:
    uniform_frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    poly, grasp, meta = extract_contour_grasp_in_roi(uniform_frame, (40, 50, 120, 90))
    assert meta["contour_method"] == "bbox_fallback"
    assert poly.shape == (4, 2)
    assert grasp is not None

    tiny_poly, tiny_grasp, tiny_meta = extract_contour_grasp_in_roi(uniform_frame, (10, 10, 14, 15))
    assert tiny_meta["contour_method"] == "bbox_fallback"
    assert tiny_poly.shape == (4, 2)
    assert tiny_grasp is not None


def _horizontal_grasp(
    cx: float, cy: float, width_px: float = 24.0, length_px: float = 90.0
) -> tuple[np.ndarray, GraspCandidate]:
    hl, hw = length_px * 0.5, width_px * 0.5
    poly = np.array(
        [[cx - hl, cy - hw], [cx + hl, cy - hw], [cx + hl, cy + hw], [cx - hl, cy + hw]],
        dtype=np.float32,
    )
    grasp = compute_grasp_from_polygon(poly)
    assert grasp is not None
    return poly, grasp


def test_rank_grasp_candidates_prioritizes_isolated_unoccluded_vape() -> None:
    # Vape A: highest confidence (0.92), but occluded by a pen lying across it
    poly_a, grasp_a = _horizontal_grasp(120.0, 90.0)
    vape_a = Detection(
        label="vape",
        confidence=0.92,
        bounding_box=(75, 78, 165, 102),
        polygon=poly_a,
        grasp=grasp_a,
        metadata={"id": "A", "vlm": {"verified": True, "is_vape": True, "is_occluded": True}},
    )

    # Vape B: high confidence (0.90), horizontal -> gripper_angle=+-90 deg (jaws at y+-27.6)
    poly_b, grasp_b = _horizontal_grasp(120.0, 220.0, width_px=24.0)
    vape_b = Detection(
        label="vape",
        confidence=0.90,
        bounding_box=(75, 208, 165, 232),
        polygon=poly_b,
        grasp=grasp_b,
        metadata={"id": "B", "vlm": {"verified": True, "is_vape": True, "is_occluded": False}},
    )
    # Neighbor clutter placed right at Vape B's bottom jaw tip (y ~ 247.6)
    clutter_poly = np.array(
        [[95, 240], [145, 240], [145, 265], [95, 265]],
        dtype=np.float32,
    )
    neighbor_clutter = Detection(
        label="non_vape:ballpoint_pen",
        confidence=0.88,
        bounding_box=(95, 240, 145, 265),
        polygon=clutter_poly,
    )

    # Vape C: slightly lower confidence (0.84), but isolated, unoccluded, clear jaws
    poly_c, grasp_c = _horizontal_grasp(360.0, 180.0)
    vape_c = Detection(
        label="vape",
        confidence=0.84,
        bounding_box=(315, 168, 405, 192),
        polygon=poly_c,
        grasp=grasp_c,
        metadata={"id": "C", "vlm": {"verified": True, "is_vape": True, "is_occluded": False}},
    )

    ranked = rank_grasp_candidates(
        [vape_a, vape_b, neighbor_clutter, vape_c],
        frame_shape=(360, 500),
    )

    assert len(ranked) == 4
    first, second, third, fourth = ranked

    assert first.metadata["id"] == "C"
    assert first.metadata["robotics"]["pick_rank"] == 1
    assert first.metadata["robotics"]["clearance_ok"] is True
    assert first.metadata["robotics"]["blocking_reasons"] == []

    blocked_ids = {second.metadata["id"], third.metadata["id"]}
    assert blocked_ids == {"A", "B"}
    assert second.metadata["robotics"]["pick_rank"] == 2
    assert second.metadata["robotics"]["clearance_ok"] is False
    assert third.metadata["robotics"]["pick_rank"] == 3
    assert third.metadata["robotics"]["clearance_ok"] is False

    reasons_by_id = {
        det.metadata["id"]: det.metadata["robotics"]["blocking_reasons"] for det in (second, third)
    }
    assert "occluded_by_object" in reasons_by_id["A"]
    assert "jaw_collision_neighbor" in reasons_by_id["B"]

    assert fourth.label == "non_vape:ballpoint_pen"
    assert fourth.metadata["robotics"]["pick_rank"] is None


def test_hybrid_vlm_detector_end_to_end_two_call_flow() -> None:
    height, width = 400, 600
    frame = np.full((height, width, 3), (122, 124, 126), dtype=np.uint8)

    # 1. Known Stage-1 vape on left side
    known_poly, known_grasp = _horizontal_grasp(120.0, 120.0, width_px=26.0, length_px=96.0)
    cv2.fillPoly(
        frame,
        [np.round(known_poly).astype(np.int32).reshape(-1, 1, 2)],
        (60, 180, 240),
    )
    stage1_vape = Detection(
        label="vape",
        confidence=0.55,
        bounding_box=(72, 107, 168, 133),
        polygon=known_poly,
        grasp=known_grasp,
    )

    # 2. Stage-1 false positive on AUKEY cardboard box
    aukey_fp = Detection(
        label="vape",
        confidence=0.62,
        bounding_box=(60, 260, 200, 340),
    )

    # 3. Missed rotated shiny box puff on right side (YOLO missed completely!)
    missed_pts = _draw_rotated_puff_bar(
        frame,
        center=(430.0, 210.0),
        length_px=126.0,
        width_px=38.0,
        angle_deg=-32.0,
        color_bgr=(215, 85, 30),
    )
    mx1 = int(np.floor(missed_pts[:, 0].min())) - 14
    my1 = int(np.floor(missed_pts[:, 1].min())) - 14
    mx2 = int(np.ceil(missed_pts[:, 0].max())) + 14
    my2 = int(np.ceil(missed_pts[:, 1].max())) + 14

    call1_batch_json = json.dumps(
        [
            {
                "index": 0,
                "is_vape": True,
                "detected_object": "disposable_vape",
                "confidence": 0.92,
                "hazard_note": "Li-ion + e-liquid fire hazard",
            },
            {
                "index": 1,
                "is_vape": False,
                "detected_object": "printed_packaging_graphic",
                "confidence": 0.96,
                "hazard_note": "2D printed webcam bar on AUKEY cardboard box",
            },
        ]
    )

    # Normalized 0..1000 [ymin, xmin, ymax, xmax] for missed puff + duplicate of stage1_vape
    missed_box_norm = [
        round(my1 / height * 1000),
        round(mx1 / width * 1000),
        round(my2 / height * 1000),
        round(mx2 / width * 1000),
    ]
    dup_box_norm = [
        round(105 / height * 1000),
        round(70 / width * 1000),
        round(135 / height * 1000),
        round(170 / width * 1000),
    ]
    call2_discovery_json = json.dumps(
        [
            {
                "box_2d": missed_box_norm,
                "detected_object": "metallic_box_puff",
                "confidence": 0.94,
                "is_occluded": False,
                "occlusion_note": "",
                "hazard_note": "Embedded Li-ion cell in Lost Mary box puff",
            },
            {
                "box_2d": dup_box_norm,
                "detected_object": "disposable_vape",
                "confidence": 0.89,
                "is_occluded": False,
                "occlusion_note": "",
                "hazard_note": "Duplicate of already found vape",
            },
        ]
    )

    client = _FakeGenAiClient([call1_batch_json, call2_discovery_json])
    detector = HybridVlmDetector(
        base_detector=_StubDetector([stage1_vape, aukey_fp], mm_per_px=0.5),
        reject_non_vapes=True,
        discovery_trigger="always",
        client=client,
    )

    detections = detector.detect(frame)

    assert len(client.models.calls) == 2
    assert len(detections) == 2
    assert all(det.label == "vape" for det in detections)
    assert [det.metadata["robotics"]["pick_rank"] for det in detections] == [1, 2]

    discovered = [det for det in detections if det.metadata.get("source") == "vlm_scene_discovery"]
    assert len(discovered) == 1
    disc_det = discovered[0]
    assert disc_det.metadata["vlm"]["discovered"] is True
    assert disc_det.metadata["vlm"]["detected_object"] == "metallic_box_puff"
    assert disc_det.metadata["contour_method"] == "lab_foreground"
    assert disc_det.grasp is not None
    assert math.isfinite(disc_det.grasp.object_angle_deg)
    assert abs(disc_det.grasp.object_angle_deg - (-32.0)) <= 6.0
    assert "gripper_opening_mm" in disc_det.metadata


def test_hybrid_discovery_trigger_when_empty_behavior() -> None:
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    vape_det = Detection(label="vape", confidence=0.50, bounding_box=(20, 30, 100, 60))

    # Case 1: Stage 1 confirms a vape -> Call 2 (scene discovery) is skipped
    client_has_vape = _FakeGenAiClient(
        [
            json.dumps(
                [
                    {
                        "index": 0,
                        "is_vape": True,
                        "detected_object": "disposable_vape",
                        "confidence": 0.90,
                    }
                ]
            )
        ]
    )
    det_skip = HybridVlmDetector(
        base_detector=_StubDetector([vape_det]),
        discovery_trigger="when_empty",
        client=client_has_vape,
    )
    res_skip = det_skip.detect(frame)
    assert len(client_has_vape.models.calls) == 1
    assert len(res_skip) == 1
    assert res_skip[0].metadata["robotics"]["pick_rank"] == 1

    # Case 2: Stage 1 rejects lookalike -> 0 confirmed vapes -> Call 2 runs and discovers vape
    client_empty = _FakeGenAiClient(
        [
            json.dumps(
                [
                    {
                        "index": 0,
                        "is_vape": False,
                        "detected_object": "aa_battery",
                        "confidence": 0.95,
                    }
                ]
            ),
            json.dumps(
                [
                    {
                        "box_2d": [500, 450, 700, 800],
                        "detected_object": "upright_vape",
                        "confidence": 0.87,
                        "is_occluded": False,
                    }
                ]
            ),
        ]
    )
    det_run = HybridVlmDetector(
        base_detector=_StubDetector([vape_det]),
        discovery_trigger="when_empty",
        reject_non_vapes=True,
        client=client_empty,
    )
    res_run = det_run.detect(frame)
    assert len(client_empty.models.calls) == 2
    assert len(res_run) == 1
    assert res_run[0].metadata["source"] == "vlm_scene_discovery"
    assert res_run[0].metadata["robotics"]["pick_rank"] == 1
    assert isinstance(
        SceneDiscoveryCandidate(
            bounding_box=res_run[0].bounding_box,
            detected_object="upright_vape",
            confidence=0.87,
        ),
        SceneDiscoveryCandidate,
    )


def test_hybrid_init_raises_import_error_when_google_genai_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="uv sync --extra vlm"):
        HybridVlmDetector(base_detector=_StubDetector([]), client=None)


def test_hybrid_missing_api_key_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_google_genai_spec(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    poly, grasp = _horizontal_grasp(90.0, 80.0)
    raw_det = Detection(
        label="vape",
        confidence=0.52,
        bounding_box=(45, 68, 135, 92),
        polygon=poly,
        grasp=grasp,
    )
    offline_hybrid = HybridVlmDetector(
        base_detector=_StubDetector([raw_det]),
        api_key=None,
        client=None,
    )
    results = offline_hybrid.detect(np.full((180, 240, 3), 110, dtype=np.uint8))

    assert len(results) == 1
    assert results[0].label == "vape"
    assert results[0].metadata["vlm"] == {"verified": False, "reason": "missing_api_key"}
    assert results[0].metadata["robotics"]["pick_rank"] == 1

    with pytest.raises(ValueError, match="discovery_trigger"):
        HybridVlmDetector(
            base_detector=_StubDetector([]),
            discovery_trigger="invalid",  # type: ignore[arg-type]
            client=_FakeGenAiClient([]),
        )
