from io import BytesIO
from unittest.mock import MagicMock

import requests
from PIL import Image

from leukerbad.data.openimages import CLASSES, _collect_candidates


def _response(lines: list[str]) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.iter_lines.return_value = [line.encode() for line in lines]
    return response


def test_candidate_selection_requires_positive_label_and_downloadable_id(monkeypatch) -> None:
    monkeypatch.setattr(
        "leukerbad.data.openimages.requests.get",
        lambda *args, **kwargs: _response(
            [
                "ImageID,Source,LabelName,Confidence",
                "good-eraser,verification,/eraser,1",
                "unavailable-eraser,verification,/eraser,1",
                "negative-pen,verification,/pen,0",
                "good-pen,verification,/pen,1",
                "existing-pen,verification,/pen,1",
            ]
        ),
    )
    deficits = {name: 0 for name in CLASSES}
    deficits.update(eraser=1, pen=1)

    candidates, fallbacks = _collect_candidates(
        "https://example.test/labels.csv",
        {"eraser": "/eraser", "pen": "/pen"},
        {"good-eraser", "good-pen"},
        {"existing-pen"},
        deficits,
    )

    assert candidates["eraser"] == {"good-eraser"}
    assert candidates["pen"] == {"good-pen"}
    assert all(not values for values in fallbacks.values())


def test_candidate_selection_keeps_unavailable_ids_as_fallbacks(monkeypatch) -> None:
    monkeypatch.setattr(
        "leukerbad.data.openimages.requests.get",
        lambda *args, **kwargs: _response(
            ["ImageID,Source,LabelName,Confidence", "outside-bucket,verification,/torch,1"]
        ),
    )
    deficits = {name: 0 for name in CLASSES}
    deficits["torch"] = 1

    candidates, fallbacks = _collect_candidates(
        "https://example.test/labels.csv",
        {"torch": "/torch"},
        set(),
        set(),
        deficits,
    )

    assert candidates["torch"] == set()
    assert fallbacks["torch"] == {"outside-bucket"}


def test_download_retries_and_validates_image_before_rename(tmp_path, monkeypatch) -> None:
    from leukerbad.data import openimages

    image_data = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(image_data, format="JPEG")

    class Response:
        def __init__(self, content: bytes = b"", fail: bool = False) -> None:
            self.content = content
            self.fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            if self.fail:
                raise requests.HTTPError("temporary error")

        def iter_content(self, _chunk_size):
            yield self.content

    calls = iter([Response(fail=True), Response(image_data.getvalue())])
    monkeypatch.setattr(openimages.requests, "get", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(openimages.time, "sleep", lambda _: None)
    path = tmp_path / "image.jpg"

    openimages._download_image("https://example.test/image.jpg", path)

    with Image.open(path) as image:
        image.verify()
    assert not path.with_suffix(".jpg.part").exists()


def test_complete_existing_subset_is_enriched_without_rescanning_labels(tmp_path, monkeypatch) -> None:
    import json

    from leukerbad.data import openimages

    data = tmp_path / "public_data"
    metadata_dir = data / "raw/open_images/metadata"
    metadata_dir.mkdir(parents=True)
    image_dir = data / "raw/open_images/images/eraser"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "image-id.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    classes = metadata_dir / "oidv7-class-descriptions-boxable.csv"
    classes.write_text("LabelName,DisplayName\n/eraser,Eraser\n", encoding="utf-8")
    metadata_header = (
        "ImageID,Subset,OriginalURL,OriginalLandingURL,License,Author,Title,Thumbnail300KURL\n"
    )
    metadata_row = (
        "image-id,validation,https://source.test/image,https://landing.test/image,"
        "https://creativecommons.org/licenses/by/2.0/,Author,Title,https://thumb.test/image\n"
    )
    validation_metadata = metadata_dir / "validation-images-with-rotation.csv"
    validation_metadata.write_text(metadata_header + metadata_row, encoding="utf-8")
    train_metadata = metadata_dir / "train-images-boxable-with-rotation.csv"
    train_metadata.write_text(metadata_header, encoding="utf-8")

    monkeypatch.setattr(openimages, "CLASSES", ["eraser"])
    monkeypatch.setattr(openimages, "SELECTION_ORDER", ["eraser"])
    monkeypatch.setattr(openimages, "PER_CLASS", 1)
    monkeypatch.setattr(openimages, "_download_file", lambda _url, _path: None)
    monkeypatch.setattr(openimages, "data_root", lambda: data)
    monkeypatch.setattr(openimages, "_load_classes", lambda _path: {"eraser": "/eraser"})
    monkeypatch.setattr(openimages, "_collect_candidates", lambda *args: (_ for _ in ()).throw(AssertionError()))

    result = openimages.download_open_images({"id": "open_images_test", "raw_dir": "public_data/raw/open_images", "license": "CC BY 2.0"})

    assert result["status"] == "downloaded"
    assert result["per_class_counts"] == {"eraser": 1}
    record = json.loads((data / "manifests/open_images_negatives.jsonl").read_text(encoding="utf-8"))
    assert record["author"] == "Author"
    assert record["license"].endswith("by/2.0/")
    assert record["image_split"] == "validation"
    assert record["metadata_source"] == openimages.VALIDATION_METADATA_URL
