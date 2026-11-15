import json
from pathlib import Path

from leukerbad.data.common import load_sources
from leukerbad.data.curate import decision_for_class
from leukerbad.data.inspect_sources import render_report


def test_source_registry_contains_requested_sources() -> None:
    sources = load_sources()
    source_ids = {source["id"] for source in sources}
    open_images = next(source for source in sources if source["id"] == "open_images_v7_negatives")

    assert "roboflow_primary" in source_ids
    assert "kaggle_ecigarettes_vs_batteries" in source_ids
    assert "open_images_v7_negatives" in source_ids
    assert len(sources) == 4
    assert open_images["target_images"] == 600


def test_source_report_preserves_unknown_metadata() -> None:
    sources = load_sources()
    report = render_report(sources, "2026-01-01T00:00:00+00:00")

    assert "unknown" in report
    assert "CC BY 4.0" in report
    assert "e-cigarets-vs-batteries" in report


def test_source_report_shows_open_images_partial_progress(tmp_path: Path, monkeypatch) -> None:
    import leukerbad.data.inspect_sources as inspector

    monkeypatch.setattr(inspector, "data_root", lambda: tmp_path)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "download_open_images_v7_negatives.json").write_text(
        json.dumps({"status": "partial", "downloaded": 371, "requested": 600}),
        encoding="utf-8",
    )
    source = next(source for source in load_sources() if source["kind"] == "open_images")

    report = inspector.render_report([source], "2026-01-01T00:00:00+00:00")

    assert "371/600 target" in report
    assert "partial: 371/600 images" in report
    assert "n/a (image labels used for selection)" in report


def test_source_registry_is_valid_json() -> None:
    source_path = Path("public_data/dataset_sources.json")
    parsed = json.loads(source_path.read_text(encoding="utf-8"))

    assert isinstance(parsed["sources"], list)


def test_class_mapping_keeps_ambiguous_classes_out_of_positive() -> None:
    assert decision_for_class("vape")[0] == "positive"
    assert decision_for_class("cigarette")[0] == "hard_negative"
    assert decision_for_class("Battery")[0] == "manual_review"
