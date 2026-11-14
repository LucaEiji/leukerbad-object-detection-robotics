import pytest

from leukerbad.data.standardize import coco_boxes, filter_yolo


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
