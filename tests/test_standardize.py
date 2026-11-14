import json

import pytest
import yaml
from PIL import Image

from leukerbad.data.manual_filter import (
    EXPLICIT_ADDITIONAL,
    VAPE_AREA_THRESHOLD,
    manual_exclusion_reason,
)
from leukerbad.data.standardize import EXCLUDED_KAGGLE, YOLO_SOURCES, build, coco_boxes, filter_yolo


def test_filter_keeps_only_target_boxes_and_retains_negative_result():
    names = ["battery", "Vape", "possiblyVape", "vape", "ecigarette"]
    boxes, removed = filter_yolo(
        "0 .5 .5 .2 .2\n1 .4 .4 .2 .2\n2 .5 .5 .2 .2\n"
        "3 .6 .6 .1 .1\n4 .5 .5 .4 .4\n", names
    )
    assert len(boxes) == 3
    assert removed == {"battery": 1, "possiblyVape": 1}
    assert filter_yolo("0 .5 .5 .2 .2", names) == ([], {"battery": 1})
    assert filter_yolo("", names) == ([], {})
    assert filter_yolo("1 .1 .2 .5 .2 .5 .6 .1 .6", names)[0][0] == pytest.approx(
        [.3, .4, .4, .4]
    )


def test_coco_conversion_filters_battery_and_checks_box_edges():
    boxes, removed = coco_boxes([
        {"category_name": "ecigarette", "bbox": [20, 10, 40, 20]},
        {"category_name": "battery", "bbox": [0, 0, 10, 10]},
    ], 100, 50)
    assert boxes == [[.4, .4, .4, .4]]
    assert removed == {"battery": 1}
    with pytest.raises(ValueError):
        coco_boxes([{"category_name": "ecigarette", "bbox": [90, 0, 20, 10]}], 100, 50)


def test_historical_cleanup_is_source_specific_and_uses_original_polygon_area():
    names = ["cigarette", "vape"]
    assert manual_exclusion_reason(
        "roboflow_primary", "cig.jpg", "0 .5 .5 .2 .2", names
    ) == "primary_cigarette_only"
    for width in (VAPE_AREA_THRESHOLD, VAPE_AREA_THRESHOLD / 2):
        assert manual_exclusion_reason(
            "roboflow_primary", "small.jpg", f"1 .5 .5 {width} 1", names
        ) == "primary_vape_area_at_or_below_reference_threshold"
    assert manual_exclusion_reason(
        "roboflow_primary", "large.jpg", f"1 .5 .5 {VAPE_AREA_THRESHOLD * 2} 1", names
    ) is None
    # A thin triangle has a large enclosing box but a small polygon area.
    polygon = "1 .1 .1 .9 .9 .9 .901"
    assert manual_exclusion_reason(
        "roboflow_primary", "polygon.jpg", polygon, names
    ) == "primary_vape_area_at_or_below_reference_threshold"
    assert manual_exclusion_reason("roboflow_primary", "empty.jpg", "", names) is None
    assert manual_exclusion_reason(
        "manual_photos", "small.jpg", "0 .5 .5 .001 .001", ["Vape"]
    ) is None
    for filename in EXPLICIT_ADDITIONAL:
        assert manual_exclusion_reason(
            "roboflow_additional", filename, "", []
        ) == "user_explicitly_marked_misannotated"
        assert manual_exclusion_reason("manual_photos", filename, "", []) is None


def test_fresh_export_reproduces_removals_without_ledger_and_preserves_raw(tmp_path):
    def image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100)).save(path)

    def pair(source, filename, text):
        directory = tmp_path / YOLO_SOURCES[source] / "train"
        image(directory / "images" / filename)
        label = directory / "labels" / f"{filename.rsplit('.', 1)[0]}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(text)

    for source, directory in YOLO_SOURCES.items():
        path = tmp_path / directory
        path.mkdir(parents=True)
        names = ["cigarette", "vape"] if source == "roboflow_primary" else ["Vape", "nonVape"]
        (path / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    pair("roboflow_primary", "cig.jpg", "0 .5 .5 .2 .2")
    pair("roboflow_primary", "small.jpg", "1 .5 .5 .001 .001")
    pair("roboflow_primary", "keep.jpg", "1 .5 .5 .2 .2")
    pair("roboflow_additional", EXPLICIT_ADDITIONAL[0], "0 .5 .5 .2 .2")
    pair("roboflow_additional", "negative.jpg", "1 .5 .5 .2 .2")
    pair("manual_photos", "manual.jpg", "0 .5 .5 .001 .001")
    pair("kaggle_test_manual", "example_jpg.rf.abc.jpg", "")
    kaggle = tmp_path / "public_data/raw/kaggle/extracted"
    image(kaggle / "test/images/example.jpg")
    coco_images = []
    annotations = []
    for i, name in enumerate([*sorted(EXCLUDED_KAGGLE), "valid.jpg"]):
        image(kaggle / "train/images" / name)
        coco_images.append({"id": i, "file_name": name, "width": 100, "height": 100})
        annotations.append({"id": i, "image_id": i, "category_id": 0,
                            "bbox": [100, 0, 20, 20] if name in EXCLUDED_KAGGLE else [10, 10, 20, 20]})
    (kaggle / "train_annotations.json").write_text(json.dumps({
        "images": coco_images, "annotations": annotations,
        "categories": [{"id": 0, "name": "ecigarette"}],
    }))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "export"
    summary = build(tmp_path, output)
    assert summary["totals"]["images"] == 5
    assert summary["totals"]["negative_images"] == 2
    reasons = summary["manual_removal_policy"]["applied_this_build_by_reason"]
    assert reasons == {
        "primary_cigarette_only": 1,
        "primary_vape_area_at_or_below_reference_threshold": 1,
        "user_explicitly_marked_misannotated": 1,
        "user_requested_out_of_bounds_coco_bbox_exclusion": 2,
    }
    for relative, content in before.items():
        assert (tmp_path / relative).read_bytes() == content
    assert not (tmp_path / "public_data/manifests/exclusions.jsonl").exists()
    assert (output / "labels/roboflow_additional__negative.txt").read_text() == ""
