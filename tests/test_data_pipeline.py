import json
from pathlib import Path

from leukerbad.data.common import load_sources
from leukerbad.data.curate import decision_for_class
from leukerbad.data.inspect_sources import render_report


def test_source_registry_contains_requested_sources() -> None:
    sources = load_sources()
    source_ids = {source["id"] for source in sources}

    assert "roboflow_primary" in source_ids
    assert "kaggle_ecigarettes_vs_batteries" in source_ids
    assert "open_images_v7_negatives" in source_ids
    assert len(sources) == 4


def test_source_report_preserves_unknown_metadata() -> None:
    sources = load_sources()
    report = render_report(sources, "2026-01-01T00:00:00+00:00")

    assert "unknown" in report
    assert "CC BY 4.0" in report
    assert "e-cigarets-vs-batteries" in report


def test_source_registry_is_valid_json() -> None:
    source_path = Path("public_data/dataset_sources.json")
    parsed = json.loads(source_path.read_text(encoding="utf-8"))

    assert isinstance(parsed["sources"], list)


def test_class_mapping_keeps_ambiguous_classes_out_of_positive() -> None:
    assert decision_for_class("vape")[0] == "positive"
    assert decision_for_class("cigarette")[0] == "hard_negative"
    assert decision_for_class("Battery")[0] == "manual_review"
