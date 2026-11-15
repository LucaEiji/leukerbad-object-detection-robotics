"""Multimodal VLM verification wrapper for filtering tabletop vape lookalikes."""

import importlib.util
import json
import os
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from leukerbad.detection.base import Detection, Detector

_VAPE_VS_LOOKALIKE_GUIDANCE = (
    "You are a hazmat robotics vision verifier inspecting tabletop objects for a dual-arm "
    "SO-101 waste-sorting cell. Distinguish physical 3D disposable vapes, e-cigarettes, puff "
    "bars, pod vapes, chunky box-mod puffs, and upright standing vapes (embedded Li-ion "
    "battery + e-liquid fire hazard) from benign tabletop lookalikes.\n"
    "True Vapes (is_vape: true):\n"
    "- Slim cylindrical or flat puff bars and pod vapes (contoured/duckbill mouthpiece, bottom "
    "airflow holes, LED end-cap, e-liquid window).\n"
    "- Compact chunky rounded-square metallic or matte box-mod puffs (e.g., Lost Mary, Geek Bar, "
    "Vozol) with an offset duckbill mouthpiece protruding from the top corner.\n"
    "- Upright standing end-on vapes showing a raised mouthpiece tip and physical 3D height.\n"
    "Benign Lookalikes (is_vape: false):\n"
    "1. 2D flat printed product graphics/photos on cardboard retail packaging (e.g., black "
    "webcam bar printed on a teal AUKEY box, vape photo printed on a flat ElfBar carton — look "
    "for cardboard box corners/flaps, flat specular print glare, surrounding marketing text or "
    "barcodes, and lack of 3D depth relief against the table).\n"
    "2. Pens, multi-color click pens (Bic), fountain pens (Montblanc), markers, and highlighters "
    "(Stabilo) — look for pocket clips, pen caps, nibs, writing tips, or multi-color sliders.\n"
    "3. Cylindrical batteries (AA/AAA/18650 cells like Duracell/Energy) — flat or button +/- "
    "metal terminals, voltage markings, and no mouthpiece.\n"
    "4. Tactical flashlights / LED torches (CREE knurled aluminum barrel, glass reflector lens), "
    "medical asthma inhalers (Salbutamol L-shaped plastic actuator boot), AirPods/earbud cases "
    "(clamshell lid seam), lipsticks/lip balms, USB flash drives, and BIC/gas lighters."
)

_SYSTEM_INSTRUCTION = (
    f"{_VAPE_VS_LOOKALIKE_GUIDANCE}\n"
    "Respond ONLY with a single JSON object matching:\n"
    '{"is_vape": bool, "detected_object": "<snake_case_name>", '
    '"confidence": <float 0.0..1.0>, "hazard_note": "<concise hazard or lookalike note>"}'
)

_BATCH_SYSTEM_INSTRUCTION = (
    f"{_VAPE_VS_LOOKALIKE_GUIDANCE}\n"
    "You are given a full tabletop context image with numbered candidate boxes plus individual "
    "cropped ROIs (crop_0, crop_1, ...). Use the full scene to check whether a crop is merely a "
    "2D graphic printed on a cardboard box versus a real 3D object resting on the table.\n"
    "Respond ONLY with a JSON array containing one object per crop in order:\n"
    '[{"index": 0, "is_vape": bool, "detected_object": "<snake_case_name>", '
    '"confidence": <float 0.0..1.0>, "hazard_note": "<concise hazard or lookalike note>"}]'
)


@dataclass(frozen=True)
class VlmVerdict:
    """Structured multimodal verdict returned by the VLM lookalike verifier."""

    is_vape: bool
    detected_object: str
    confidence: float
    hazard_note: str


def crop_detection_roi(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    padding_ratio: float = 0.20,
) -> np.ndarray:
    """Return a padded, strictly bounds-clamped, non-empty ROI crop from ``frame``."""
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("frame must be a non-empty 2D or 3D image array")

    frame_h, frame_w = frame.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = bounding_box
    x1, x2 = min(raw_x1, raw_x2), max(raw_x1, raw_x2)
    y1, y2 = min(raw_y1, raw_y2), max(raw_y1, raw_y2)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    ratio = max(0.0, float(padding_ratio))
    pad_x = round(box_w * ratio)
    pad_y = round(box_h * ratio)

    cx1 = max(0, min(frame_w - 1, x1 - pad_x))
    cy1 = max(0, min(frame_h - 1, y1 - pad_y))
    cx2 = max(cx1 + 1, min(frame_w, x2 + pad_x))
    cy2 = max(cy1 + 1, min(frame_h, y2 + pad_y))

    return frame[cy1:cy2, cx1:cx2].copy()


def _strip_json_fences(raw_text: str) -> str:
    """Strip optional markdown code fences surrounding a JSON payload."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _dict_to_verdict(payload: dict[str, Any]) -> VlmVerdict:
    """Validate and convert a parsed JSON dict into a ``VlmVerdict``."""
    if not isinstance(payload, dict) or "is_vape" not in payload:
        raise ValueError("VLM response JSON must be an object containing 'is_vape'")

    raw_is_vape = payload["is_vape"]
    if isinstance(raw_is_vape, bool):
        is_vape = raw_is_vape
    elif isinstance(raw_is_vape, str) and raw_is_vape.lower() in {"true", "false"}:
        is_vape = raw_is_vape.lower() == "true"
    else:
        raise ValueError("VLM 'is_vape' field must be a boolean")

    raw_obj = str(payload.get("detected_object") or "").strip().lower().replace(" ", "_")
    if not raw_obj:
        raw_obj = "disposable_vape" if is_vape else "unknown_object"

    raw_conf = float(payload.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, raw_conf))

    default_note = (
        "Embedded Li-ion cell + nicotine e-liquid fire hazard"
        if is_vape
        else "Benign non-vape lookalike"
    )
    hazard_note = str(payload.get("hazard_note") or default_note).strip() or default_note

    return VlmVerdict(
        is_vape=is_vape,
        detected_object=raw_obj,
        confidence=confidence,
        hazard_note=hazard_note,
    )


def _parse_verdict_json(raw_text: str) -> VlmVerdict:
    """Parse a JSON string from Gemini into a validated ``VlmVerdict``."""
    cleaned = _strip_json_fences(raw_text)
    payload = json.loads(cleaned)
    return _dict_to_verdict(payload)


def _parse_batch_verdicts_json(
    raw_text: str,
    expected_count: int,
) -> list[VlmVerdict | None]:
    """Parse a batched JSON array from Gemini into aligned ``VlmVerdict | None`` items."""
    if expected_count <= 0:
        return []

    cleaned = _strip_json_fences(raw_text)
    payload = json.loads(cleaned)

    items: list[Any] | None = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if "is_vape" in payload and expected_count == 1:
            items = [payload]
        else:
            list_val = payload.get("verdicts") or payload.get("results") or payload.get("items")
            if isinstance(list_val, list):
                items = list_val

    if items is None:
        raise ValueError("Batched VLM response JSON must be a JSON array of verdicts")

    verdicts: list[VlmVerdict | None] = [None] * expected_count
    for pos_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_idx = item.get("index")
        if isinstance(raw_idx, (int, float)) and not isinstance(raw_idx, bool):
            idx = int(raw_idx)
        else:
            idx = pos_idx
        if 0 <= idx < expected_count:
            try:
                verdicts[idx] = _dict_to_verdict(item)
            except (ValueError, TypeError):
                verdicts[idx] = None

    return verdicts


def _has_google_genai() -> bool:
    """Return True if ``google.genai`` is importable without importing it eagerly."""
    try:
        return importlib.util.find_spec("google.genai") is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _make_genai_image_part(jpeg_bytes: bytes) -> Any:
    """Create a Gemini JPEG image part with offline fallback when ``google.genai`` is absent."""
    try:
        from google.genai import types
    except ImportError:
        return SimpleNamespace(data=jpeg_bytes, mime_type="image/jpeg")
    return types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")


def _make_genai_json_config(system_instruction: str) -> Any:
    """Create a deterministic JSON generation config with offline stub fallback."""
    try:
        from google.genai import types
    except ImportError:
        return SimpleNamespace(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.0,
        )
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=0.0,
    )


def _render_scene_context_for_batch(
    frame: np.ndarray,
    detections: list[Detection],
    max_dim: int = 960,
) -> np.ndarray:
    """Downscale ``frame`` to ``max_dim`` and annotate numbered candidate boxes."""
    frame_h, frame_w = frame.shape[:2]
    peak = max(frame_h, frame_w)
    if peak > max_dim:
        scale = max_dim / float(peak)
        new_w = max(1, round(frame_w * scale))
        new_h = max(1, round(frame_h * scale))
        canvas = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        canvas = frame.copy()

    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    ch, cw = canvas.shape[:2]
    for idx, det in enumerate(detections):
        rx1, ry1, rx2, ry2 = det.bounding_box
        x1 = max(0, min(cw - 1, round(min(rx1, rx2) * scale)))
        y1 = max(0, min(ch - 1, round(min(ry1, ry2) * scale)))
        x2 = max(x1 + 1, min(cw, round(max(rx1, rx2) * scale)))
        y2 = max(y1 + 1, min(ch, round(max(ry1, ry2) * scale)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2, lineType=cv2.LINE_AA)
        tag = f"crop_{idx}"
        ty = y1 - 5 if y1 >= 18 else min(ch - 4, y2 + 14)
        cv2.putText(
            canvas,
            tag,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return canvas


class VlmVerifiedDetector:
    """Two-stage detector that verifies ambiguous YOLO crops with Gemini VLM."""

    def __init__(
        self,
        base_detector: Detector,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        min_conf_to_verify: float = 0.20,
        max_conf_to_verify: float = 0.85,
        padding_ratio: float = 0.20,
        reject_non_vapes: bool = False,
        batch_verify: bool = False,
        client: Any | None = None,
        *,
        _skip_genai_check: bool = False,
    ) -> None:
        if not (0.0 <= min_conf_to_verify <= max_conf_to_verify <= 1.0):
            raise ValueError(
                "Confidence bounds must satisfy "
                "0.0 <= min_conf_to_verify <= max_conf_to_verify <= 1.0"
            )
        if not _skip_genai_check and client is None and not _has_google_genai():
            raise ImportError(
                "google-genai is required for VlmVerifiedDetector. "
                "Install with: uv sync --extra vlm"
            )

        self.base_detector = base_detector
        self.model_name = model_name
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self.min_conf_to_verify = min_conf_to_verify
        self.max_conf_to_verify = max_conf_to_verify
        self.padding_ratio = padding_ratio
        self.reject_non_vapes = reject_non_vapes
        self.batch_verify = bool(batch_verify)
        self._client = client

    def _get_or_create_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for VlmVerifiedDetector. "
                "Install with: uv sync --extra vlm"
            ) from exc

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _verify_crop(self, jpeg_bytes: bytes, det: Detection) -> VlmVerdict:
        client = self._get_or_create_client()
        image_part = _make_genai_image_part(jpeg_bytes)
        config = _make_genai_json_config(_SYSTEM_INSTRUCTION)

        user_prompt = (
            f"Stage-1 detector proposed label='{det.label}' (confidence={det.confidence:.2f}). "
            "Inspect mouthpieces, offset box-mod duckbill tips, 3D depth relief vs 2D printed "
            "cardboard packaging graphics (AUKEY/ElfBar boxes), pen clips/nibs, +/- battery "
            "terminals, flashlight lenses, inhaler boots, and seams to determine whether this "
            "object is a real physical 3D vape/puff or a benign lookalike."
        )
        response = client.models.generate_content(
            model=self.model_name,
            contents=[image_part, user_prompt],
            config=config,
        )
        raw_text = getattr(response, "text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Empty response text from VLM client")
        return _parse_verdict_json(raw_text)

    def verify_crops_batch(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[VlmVerdict | None]:
        """Verify multiple candidate detections in a single Gemini multimodal call."""
        if not detections:
            return []

        client = self._get_or_create_client()
        scene_img = _render_scene_context_for_batch(frame, detections, max_dim=960)
        scene_ok, scene_buf = cv2.imencode(".jpg", scene_img)
        if not scene_ok:
            raise RuntimeError("cv2.imencode failed for scene context image")

        contents: list[Any] = [
            _make_genai_image_part(scene_buf.tobytes()),
            (
                "Full tabletop scene context with numbered candidate boxes (crop_0, crop_1, ...). "
                "Check surrounding cardboard packaging edges, print glare, and 3D relief."
            ),
        ]

        for idx, det in enumerate(detections):
            crop = crop_detection_roi(
                frame,
                det.bounding_box,
                padding_ratio=self.padding_ratio,
            )
            encoded_ok, encoded_buf = cv2.imencode(".jpg", crop)
            if not encoded_ok:
                raise RuntimeError(f"cv2.imencode failed for ROI crop_{idx}")
            contents.append(_make_genai_image_part(encoded_buf.tobytes()))
            contents.append(
                f"crop_{idx}: stage-1 label='{det.label}', "
                f"confidence={det.confidence:.2f}, bbox={det.bounding_box}."
            )

        contents.append(
            f"Verify all {len(detections)} candidate crops (index 0..{len(detections) - 1}). "
            "Confirm physical 3D disposable vapes, pod vapes, chunky metallic/matte box-mod "
            "puffs, and upright standing vapes (is_vape=true). Reject 2D graphics printed on "
            "cardboard boxes (e.g., AUKEY webcam box, ElfBar retail carton), pens/highlighters, "
            "AA/AAA batteries, tactical flashlights, asthma inhalers, AirPods cases, lighters, "
            "lipsticks, and USB sticks (is_vape=false). Return a JSON array of verdict objects."
        )

        config = _make_genai_json_config(_BATCH_SYSTEM_INSTRUCTION)
        response = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        raw_text = getattr(response, "text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Empty response text from VLM client")
        return _parse_batch_verdicts_json(raw_text, len(detections))

    def _apply_verdict(self, det: Detection, verdict: VlmVerdict) -> Detection | None:
        vlm_meta = {
            "verified": True,
            "model": self.model_name,
            "is_vape": verdict.is_vape,
            "detected_object": verdict.detected_object,
            "confidence": verdict.confidence,
            "hazard_note": verdict.hazard_note,
        }
        if verdict.is_vape:
            blended_conf = round(0.4 * det.confidence + 0.6 * verdict.confidence, 4)
            return replace(
                det,
                label="vape",
                confidence=blended_conf,
                metadata={
                    **det.metadata,
                    "raw_yolo_label": det.label,
                    "vlm": vlm_meta,
                },
            )
        if self.reject_non_vapes:
            return None
        return replace(
            det,
            label=f"non_vape:{verdict.detected_object}",
            confidence=round(verdict.confidence, 4),
            metadata={
                **det.metadata,
                "raw_yolo_label": det.label,
                "vlm": vlm_meta,
            },
        )

    def _detect_batched(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        has_credentials = self._client is not None or bool(self.api_key)
        slot_results: list[Detection | None] = [None] * len(detections)
        in_band_indices: list[int] = []
        in_band_dets: list[Detection] = []

        for idx, det in enumerate(detections):
            if det.confidence < self.min_conf_to_verify or det.confidence > self.max_conf_to_verify:
                slot_results[idx] = replace(
                    det,
                    metadata={
                        **det.metadata,
                        "vlm": {"verified": False, "reason": "outside_conf_band"},
                    },
                )
            elif not has_credentials:
                slot_results[idx] = replace(
                    det,
                    metadata={
                        **det.metadata,
                        "vlm": {"verified": False, "reason": "missing_api_key"},
                    },
                )
            else:
                in_band_indices.append(idx)
                in_band_dets.append(det)

        if in_band_dets:
            try:
                verdicts = self.verify_crops_batch(frame, in_band_dets)
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001
                err_reason = f"error:{exc.__class__.__name__}"
                for slot_idx, det in zip(in_band_indices, in_band_dets, strict=True):
                    slot_results[slot_idx] = replace(
                        det,
                        metadata={
                            **det.metadata,
                            "vlm": {"verified": False, "reason": err_reason},
                        },
                    )
            else:
                for pos, (slot_idx, det) in enumerate(
                    zip(in_band_indices, in_band_dets, strict=True)
                ):
                    verdict = verdicts[pos] if pos < len(verdicts) else None
                    if verdict is None:
                        slot_results[slot_idx] = replace(
                            det,
                            metadata={
                                **det.metadata,
                                "vlm": {"verified": False, "reason": "missing_batch_verdict"},
                            },
                        )
                    else:
                        slot_results[slot_idx] = self._apply_verdict(det, verdict)

        return [item for item in slot_results if item is not None]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run stage-1 detection and verify in-band candidates via Gemini VLM."""
        detections = self.base_detector.detect(frame)
        if self.batch_verify:
            return self._detect_batched(frame, detections)

        verified_detections: list[Detection] = []
        for det in detections:
            if det.confidence < self.min_conf_to_verify or det.confidence > self.max_conf_to_verify:
                verified_detections.append(
                    replace(
                        det,
                        metadata={
                            **det.metadata,
                            "vlm": {"verified": False, "reason": "outside_conf_band"},
                        },
                    )
                )
                continue

            if self._client is None and not self.api_key:
                verified_detections.append(
                    replace(
                        det,
                        metadata={
                            **det.metadata,
                            "vlm": {"verified": False, "reason": "missing_api_key"},
                        },
                    )
                )
                continue

            try:
                crop = crop_detection_roi(
                    frame,
                    det.bounding_box,
                    padding_ratio=self.padding_ratio,
                )
                encoded_ok, encoded_buf = cv2.imencode(".jpg", crop)
                if not encoded_ok:
                    raise RuntimeError("cv2.imencode failed for ROI crop")
                verdict = self._verify_crop(encoded_buf.tobytes(), det)
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001
                verified_detections.append(
                    replace(
                        det,
                        metadata={
                            **det.metadata,
                            "vlm": {
                                "verified": False,
                                "reason": f"error:{exc.__class__.__name__}",
                            },
                        },
                    )
                )
                continue

            updated = self._apply_verdict(det, verdict)
            if updated is not None:
                verified_detections.append(updated)

        return verified_detections
