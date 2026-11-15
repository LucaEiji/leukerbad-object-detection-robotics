"""Unit tests for the `leukerbad-detect` CLI entry point."""

import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from leukerbad import cli
from leukerbad.detection.base import Detection
from leukerbad.detection.geometry import compute_grasp_from_polygon


def test_cli_bare_invocation_prints_status_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main([])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Leukerbad" in captured.out
    assert "--synthetic-demo" in captured.out


def test_cli_synthetic_demo_saves_annotated_image(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "outputs" / "synthetic_annotated.png"
    rc = cli.main(["--synthetic-demo", "--mm-per-px", "0.5", "--save", str(out_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out_path.is_file()
    saved = cv2.imread(str(out_path))
    assert saved is not None
    assert saved.shape == (480, 640, 3)
    assert "Detected 3 object(s):" in captured.out
    assert "disposable_vape" in captured.out
    assert "opening=" in captured.out


def test_cli_synthetic_demo_only_vapes_filters_distractors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["--synthetic-demo", "--only-vapes"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Detected 2 object(s):" in captured.out
    assert "non_vape_pen" not in captured.out


def test_cli_image_mode_with_mocked_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_img = tmp_path / "tabletop.png"
    output_img = tmp_path / "annotated.png"
    raw = np.full((240, 320, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(input_img), raw)

    recorded_kwargs: dict[str, Any] = {}

    class _StubDetector:
        def __init__(self, **kwargs: Any) -> None:
            recorded_kwargs.update(kwargs)

        def detect(self, frame: np.ndarray) -> list[Detection]:
            assert frame.shape == (240, 320, 3)
            poly = np.array(
                [[80.0, 90.0], [200.0, 90.0], [200.0, 118.0], [80.0, 118.0]],
                dtype=np.float32,
            )
            return [
                Detection(
                    label="vape",
                    confidence=0.91,
                    bounding_box=(80, 90, 200, 118),
                    polygon=poly,
                    grasp=compute_grasp_from_polygon(poly),
                )
            ]

    monkeypatch.setattr(cli, "YoloSegDetector", _StubDetector)

    rc = cli.main(
        [
            "--image",
            str(input_img),
            "--save",
            str(output_img),
            "--weights",
            "custom.pt",
            "--conf",
            "0.4",
            "--labels",
            "vape, disposable_vape",
            "--mm-per-px",
            "0.6",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert output_img.is_file()
    assert recorded_kwargs["weights"] == "custom.pt"
    assert recorded_kwargs["conf_threshold"] == pytest.approx(0.4)
    assert recorded_kwargs["allowed_labels"] == {"vape", "disposable_vape"}
    assert recorded_kwargs["mm_per_px"] == pytest.approx(0.6)
    assert "vape (0.91)" in captured.out


def test_cli_missing_image_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["--image", str(tmp_path / "does_not_exist.png")])
    captured = capsys.readouterr()

    assert rc == 1
    assert "not found" in captured.err


def test_cli_gemini_verifier_wraps_detector_lazily(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    wrapped_calls: list[dict[str, Any]] = []

    class _FakeVlmVerifiedDetector:
        def __init__(
            self,
            base_detector: Any,
            min_conf_to_verify: float = 0.20,
            max_conf_to_verify: float = 0.85,
            reject_non_vapes: bool = False,
        ) -> None:
            self.base_detector = base_detector
            wrapped_calls.append(
                {
                    "min_conf_to_verify": min_conf_to_verify,
                    "max_conf_to_verify": max_conf_to_verify,
                    "reject_non_vapes": reject_non_vapes,
                }
            )

        def detect(self, frame: np.ndarray) -> list[Detection]:
            return self.base_detector.detect(frame)[:1]

    fake_module = types.ModuleType("leukerbad.detection.vlm_verifier")
    fake_module.VlmVerifiedDetector = _FakeVlmVerifiedDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "leukerbad.detection.vlm_verifier", fake_module)

    rc = cli.main(
        [
            "--synthetic-demo",
            "--verifier",
            "gemini",
            "--only-vapes",
            "--verify-min",
            "0.30",
            "--verify-max",
            "0.75",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert wrapped_calls == [
        {
            "min_conf_to_verify": pytest.approx(0.30),
            "max_conf_to_verify": pytest.approx(0.75),
            "reject_non_vapes": True,
        }
    ]
    assert "Detected 1 object(s):" in captured.out


def test_cli_gemini_verifier_missing_api_key_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    rc = cli.main(["--synthetic-demo", "--verifier", "gemini"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "GEMINI_API_KEY" in captured.err
    assert "GOOGLE_API_KEY" in captured.err


def test_cli_gemini_verifier_missing_vlm_extra_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class _MissingExtraDetector:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "google-genai is required for VlmVerifiedDetector. "
                "Install with: uv sync --extra vlm"
            )

    fake_module = types.ModuleType("leukerbad.detection.vlm_verifier")
    fake_module.VlmVerifiedDetector = _MissingExtraDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "leukerbad.detection.vlm_verifier", fake_module)

    rc = cli.main(["--synthetic-demo", "--verifier", "gemini"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "uv sync --extra vlm" in captured.err


def test_cli_only_vapes_with_gemini_verifier_filters_unverified_non_vape(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class _FakeVerifierWithOutOfBandNonVape:
        def __init__(self, base_detector: Any, **kwargs: Any) -> None:
            self.base_detector = base_detector
            self.kwargs = kwargs

        def detect(self, frame: np.ndarray) -> list[Detection]:
            del frame
            return [
                Detection(
                    label="vape",
                    confidence=0.88,
                    bounding_box=(40, 50, 140, 80),
                    metadata={"vlm": {"verified": False, "reason": "outside_conf_band"}},
                ),
                Detection(
                    label="non_vape_pen",
                    confidence=0.92,
                    bounding_box=(200, 120, 320, 150),
                    metadata={"vlm": {"verified": False, "reason": "outside_conf_band"}},
                ),
            ]

    fake_module = types.ModuleType("leukerbad.detection.vlm_verifier")
    fake_module.VlmVerifiedDetector = _FakeVerifierWithOutOfBandNonVape  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "leukerbad.detection.vlm_verifier", fake_module)

    rc = cli.main(["--synthetic-demo", "--verifier", "gemini", "--only-vapes"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Detected 1 object(s):" in captured.out
    assert "vape (0.88)" in captured.out
    assert "non_vape_pen" not in captured.out


def test_format_camera_snapshot_path_increments_without_overwriting() -> None:
    assert cli._format_camera_snapshot_path(None, 1) == Path("leukerbad_snapshot_001.png")
    assert cli._format_camera_snapshot_path(None, 2) == Path("leukerbad_snapshot_002.png")
    assert cli._format_camera_snapshot_path(Path("runs/out.png"), 1) == Path("runs/out_001.png")
    assert cli._format_camera_snapshot_path(Path("runs/out.png"), 2) == Path("runs/out_002.png")
    assert cli._format_camera_snapshot_path(Path("runs/out"), 3) == Path("runs/out_003.png")


def test_camera_loop_snapshots_and_vlm_throttling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    frames = [np.full((120, 160, 3), 40 + i * 10, dtype=np.uint8) for i in range(5)]
    # Keys per frame: frame 0 -> 's', frame 1 -> 's', frame 2 -> 'v', frame 3 -> none, frame 4 -> 'q'
    keys = [ord("s"), ord("s"), ord("v"), -1, ord("q")]
    vlm_calls = 0
    base_calls = 0

    class _FakeCap:
        def __init__(self, index: int) -> None:
            self.index = index
            self._idx = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self._idx >= len(frames):
                return False, None
            frame = frames[self._idx]
            self._idx += 1
            return True, frame

        def release(self) -> None:
            return None

    class _FakeBaseDetector:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def detect(self, frame: np.ndarray) -> list[Detection]:
            nonlocal base_calls
            del frame
            base_calls += 1
            return [
                Detection(
                    label="vape",
                    confidence=0.60,
                    bounding_box=(20, 30, 80, 70),
                )
            ]

    class _FakeVlmDetector:
        def __init__(self, base_detector: Any, **kwargs: Any) -> None:
            self.base_detector = base_detector
            self.reject_non_vapes = kwargs.get("reject_non_vapes", False)

        def detect(self, frame: np.ndarray) -> list[Detection]:
            nonlocal vlm_calls
            del frame
            vlm_calls += 1
            return [
                Detection(
                    label="non_vape:airpods_case",
                    confidence=0.95,
                    bounding_box=(22, 31, 82, 71),
                    metadata={
                        "raw_yolo_label": "vape",
                        "vlm": {
                            "verified": True,
                            "is_vape": False,
                            "detected_object": "airpods_case",
                        },
                    },
                )
            ]

    fake_module = types.ModuleType("leukerbad.detection.vlm_verifier")
    fake_module.VlmVerifiedDetector = _FakeVlmDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "leukerbad.detection.vlm_verifier", fake_module)
    monkeypatch.setattr(cli, "YoloSegDetector", _FakeBaseDetector)
    monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: keys.pop(0))

    save_base = tmp_path / "runs" / "out.png"
    rc = cli.main(
        [
            "--camera",
            "0",
            "--verifier",
            "gemini",
            "--verify-every",
            "4",
            "--save",
            str(save_base),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "'v' to force VLM verify" in captured.out
    # Frame 0 (periodic), Frame 2 ('v' forced), Frame 4 (periodic 4 % 4 == 0) -> 3 VLM calls
    # Frame 1, Frame 2 initial, Frame 3 -> 3 base_detector-only calls
    assert vlm_calls == 3
    assert base_calls == 3
    # Pressing 's' twice wrote out_001.png and out_002.png without overwriting or writing out.png
    assert (tmp_path / "runs" / "out_001.png").is_file()
    assert (tmp_path / "runs" / "out_002.png").is_file()
    assert not save_base.exists()


def test_cli_tiled_and_clahe_flags_wrap_detector(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "tiled_demo.png"
    rc = cli.main(["--synthetic-demo", "--tiled", "--clahe", "--save", str(out_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert out_path.is_file()
    assert "Detected 3 object(s):" in captured.out


def test_cli_hybrid_verifier_wires_discovery_and_pick_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    recorded: list[dict[str, Any]] = []

    class _FakeHybridDetector:
        def __init__(self, base_detector: Any, **kwargs: Any) -> None:
            self.base_detector = base_detector
            recorded.append(kwargs)

        def detect(self, frame: np.ndarray) -> list[Detection]:
            del frame
            poly = np.array(
                [[120.0, 140.0], [220.0, 140.0], [220.0, 175.0], [120.0, 175.0]],
                dtype=np.float32,
            )
            return [
                Detection(
                    label="vape",
                    confidence=0.89,
                    bounding_box=(120, 140, 220, 175),
                    polygon=poly,
                    grasp=compute_grasp_from_polygon(poly),
                    metadata={
                        "source": "vlm_scene_discovery",
                        "vlm": {
                            "verified": True,
                            "discovered": True,
                            "is_vape": True,
                            "is_occluded": False,
                        },
                        "robotics": {
                            "pick_rank": 1,
                            "grasp_priority": 0.94,
                            "clearance_ok": True,
                            "blocking_reasons": [],
                        },
                    },
                )
            ]

    fake_module = types.ModuleType("leukerbad.detection.vlm_scene")
    fake_module.HybridVlmDetector = _FakeHybridDetector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "leukerbad.detection.vlm_scene", fake_module)

    out_path = tmp_path / "hybrid_out.png"
    rc = cli.main(
        [
            "--synthetic-demo",
            "--tiled",
            "--verifier",
            "hybrid",
            "--discovery-trigger",
            "when_empty",
            "--only-vapes",
            "--mm-per-px",
            "0.45",
            "--save",
            str(out_path),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert out_path.is_file()
    assert len(recorded) == 1
    assert recorded[0]["discovery_trigger"] == "when_empty"
    assert recorded[0]["reject_non_vapes"] is True
    assert recorded[0]["mm_per_px"] == pytest.approx(0.45)
    assert "pick=#1" in captured.out


def test_cli_hybrid_verifier_missing_api_key_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    rc = cli.main(["--synthetic-demo", "--verifier", "hybrid"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "--verifier hybrid requires GEMINI_API_KEY" in captured.err
