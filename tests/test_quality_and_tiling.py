"""Offline unit tests for image quality filtering, Stage-2 dataset building, and tiled detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from leukerbad.dataset.colab_train import (
    build_parser,
    build_stage2_tabletop_dataset,
    prepare_and_train_from_drive,
)
from leukerbad.dataset.quality_filter import (
    assess_image_quality,
    compute_polygon_elongation,
    filter_dataset_split_by_quality,
    find_underrepresented_shape_stems,
)
from leukerbad.dataset.validate import _sha256_file
from leukerbad.detection.base import Detection
from leukerbad.detection.geometry import compute_grasp_from_polygon, estimate_gripper_opening_mm
from leukerbad.detection.tiled_detector import (
    TiledYoloSegDetector,
    apply_clahe_bgr,
    generate_tile_windows,
)


def _make_crisp_tabletop_image(seed: int = 7, width: int = 240, height: int = 160) -> np.ndarray:
    """Create a realistic normal-exposure tabletop image with crisp high-contrast vape shapes."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(95, 145, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(bg, (30, 35), (115, 75), (235, 220, 40), thickness=-1)
    cv2.rectangle(bg, (30, 35), (115, 75), (20, 20, 20), thickness=2)
    cv2.rectangle(bg, (140, 80), (210, 130), (30, 40, 220), thickness=-1)
    cv2.circle(bg, (175, 50), 18, (245, 245, 245), thickness=-1)
    return bg


def test_assess_image_quality_distinguishes_clean_whiteout_blackout_and_blur() -> None:
    clean_img = _make_crisp_tabletop_image(seed=11)
    clean_q = assess_image_quality(clean_img)
    assert clean_q.is_usable is True
    assert clean_q.is_whiteout is False
    assert clean_q.is_blackout is False
    assert clean_q.is_blurry is False
    assert clean_q.quality_score > 0.5

    # Pure-white overexposed augmentation (~245 luma)
    white_img = np.full((160, 240, 3), 245, dtype=np.uint8)
    white_img[40:80, 60:120] = 252
    white_q = assess_image_quality(white_img)
    assert white_q.is_whiteout is True
    assert white_q.is_usable is False
    assert white_q.quality_score < clean_q.quality_score

    # Pitch-black underexposed augmentation (~10 luma)
    black_img = np.full((160, 240, 3), 10, dtype=np.uint8)
    black_img[40:80, 60:120] = 16
    black_q = assess_image_quality(black_img)
    assert black_q.is_blackout is True
    assert black_q.is_usable is False

    # Heavily blurred low-contrast frame (normal mean luma ~128, but severe blur)
    soft_base = np.full((160, 240, 3), 115, dtype=np.uint8)
    cv2.rectangle(soft_base, (50, 40), (190, 120), (175, 175, 175), thickness=-1)
    blurry_img = cv2.GaussianBlur(soft_base, (51, 51), sigmaX=18.0)
    blurry_q = assess_image_quality(blurry_img)
    assert blurry_q.is_blurry is True
    assert blurry_q.is_whiteout is False
    assert blurry_q.is_blackout is False
    assert blurry_q.is_usable is False


def test_filter_dataset_split_by_quality_rescues_all_bad_group_and_protects_test(
    tmp_path: Path,
) -> None:
    train_img_dir = tmp_path / "images" / "train"
    train_lbl_dir = tmp_path / "labels" / "train"
    train_img_dir.mkdir(parents=True)
    train_lbl_dir.mkdir(parents=True)

    poly = "0 0.2 0.2 0.6 0.2 0.6 0.5 0.2 0.5\n"

    # Group A: 1 crisp normal + 1 whiteout + 1 blackout -> keeps only crisp normal
    samples: list[tuple[str, np.ndarray]] = [
        ("sceneA_png.rf.111111", _make_crisp_tabletop_image(1)),
        ("sceneA_png.rf.222222", np.full((120, 160, 3), 248, dtype=np.uint8)),
        ("sceneA_png.rf.333333", np.full((120, 160, 3), 8, dtype=np.uint8)),
        # Group B: all 3 variants unusable, but variant b1 is least degraded -> rescued!
        (
            "sceneB_png.rf.aaaaaa",
            cv2.GaussianBlur(_make_crisp_tabletop_image(2), (45, 45), sigmaX=14.0),
        ),
        ("sceneB_png.rf.bbbbbb", np.full((120, 160, 3), 253, dtype=np.uint8)),
        ("sceneB_png.rf.cccccc", np.full((120, 160, 3), 2, dtype=np.uint8)),
    ]
    for stem, img in samples:
        cv2.imwrite(str(train_img_dir / f"{stem}.png"), img)
        (train_lbl_dir / f"{stem}.txt").write_text(poly, encoding="utf-8")

    # Dry run must not delete anything
    dry_report = filter_dataset_split_by_quality(
        train_img_dir, train_lbl_dir, keep_best_per_group=True, dry_run=True
    )
    assert dry_report.total_checked == 6
    assert dry_report.kept_count == 2
    assert dry_report.removed_count == 4
    assert dry_report.rescued_group_best_count == 1
    assert len(list(train_img_dir.glob("*.png"))) == 6

    # Real run unlinks rejected images + labels while rescuing sceneB's best variant
    report = filter_dataset_split_by_quality(
        train_img_dir, train_lbl_dir, keep_best_per_group=True, dry_run=False
    )
    assert report.kept_count == 2
    assert report.removed_count == 4
    assert report.rescued_group_best_count == 1
    assert report.removed_whiteout >= 2
    assert report.removed_blackout >= 2

    remaining_imgs = sorted(p.stem for p in train_img_dir.glob("*.png"))
    remaining_lbls = sorted(p.stem for p in train_lbl_dir.glob("*.txt"))
    assert remaining_imgs == ["sceneA_png.rf.111111", "sceneB_png.rf.aaaaaa"]
    assert remaining_lbls == remaining_imgs

    # Pointing at ANY test split directory must immediately raise ValueError
    test_img_dir = tmp_path / "images" / "test"
    test_lbl_dir = tmp_path / "labels" / "test"
    test_img_dir.mkdir(parents=True)
    test_lbl_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="test"):
        filter_dataset_split_by_quality(test_img_dir, test_lbl_dir)

    rf_test_imgs = tmp_path / "test" / "images"
    rf_test_lbls = tmp_path / "test" / "labels"
    rf_test_imgs.mkdir(parents=True)
    rf_test_lbls.mkdir(parents=True)
    with pytest.raises(ValueError, match="test"):
        filter_dataset_split_by_quality(rf_test_imgs, rf_test_lbls)


def test_shape_elongation_and_stage2_builder_zero_leakage(tmp_path: Path) -> None:
    # Box puff polygon: 0.32 x 0.20 -> elongation = 1.6 (<= 2.2)
    box_poly = "0 0.20 0.30 0.52 0.30 0.52 0.50 0.20 0.50\n"
    # Slim pen polygon: 0.45 x 0.10 -> elongation = 4.5 (> 2.2)
    slim_poly = "0 0.20 0.40 0.65 0.40 0.65 0.50 0.20 0.50\n"

    assert compute_polygon_elongation([0.20, 0.30, 0.52, 0.30, 0.52, 0.50, 0.20, 0.50]) == (
        pytest.approx(1.6, rel=1e-2)
    )
    assert compute_polygon_elongation([0.20, 0.40, 0.65, 0.40, 0.65, 0.50, 0.20, 0.50]) == (
        pytest.approx(4.5, rel=1e-2)
    )

    probe_lbl_dir = tmp_path / "probe_labels"
    probe_lbl_dir.mkdir()
    (probe_lbl_dir / "compact_puff.txt").write_text(box_poly, encoding="utf-8")
    (probe_lbl_dir / "single_slim_pen.txt").write_text(slim_poly, encoding="utf-8")
    (probe_lbl_dir / "two_slim_clutter.txt").write_text(slim_poly + slim_poly, encoding="utf-8")
    (probe_lbl_dir / "empty_bg.txt").write_text("", encoding="utf-8")

    stems = find_underrepresented_shape_stems(probe_lbl_dir)
    assert stems == ["compact_puff", "two_slim_clutter"]

    # Build a synthetic Stage-1 normalized dataset with manifest.jsonl
    s1_root = tmp_path / "stage1_seg"
    manifest_rows: list[dict[str, object]] = []

    def _add_s1_item(
        split: str,
        stem: str,
        img: np.ndarray,
        lbl: str,
        source_id: str,
        origin_group: str,
        negative: bool = False,
        removed_battery: int = 0,
    ) -> None:
        img_dir = s1_root / "images" / split
        lbl_dir = s1_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{stem}.png"
        cv2.imwrite(str(img_dir / fname), img)
        (lbl_dir / f"{stem}.txt").write_text(lbl, encoding="utf-8")
        row: dict[str, object] = {
            "image": f"images/{fname}",
            "source_id": source_id,
            "origin_group": origin_group,
            "negative": negative,
            "split": split,
        }
        if removed_battery:
            row["removed_annotations"] = {"battery": removed_battery}
        manifest_rows.append(row)

    # manual_photos train: 1 compact box-puff (clean), 1 slim vape (clean), 1 whiteout aug (corrupt)
    _add_s1_item(
        "train",
        "manual_box_clean",
        _make_crisp_tabletop_image(101),
        box_poly,
        "manual_photos",
        "manual/g_box",
    )
    _add_s1_item(
        "train",
        "manual_box_whiteout",
        np.full((160, 240, 3), 250, dtype=np.uint8),
        box_poly,
        "manual_photos",
        "manual/g_box",
    )
    _add_s1_item(
        "train",
        "manual_slim_clean",
        _make_crisp_tabletop_image(102),
        slim_poly,
        "manual_photos",
        "manual/g_slim",
    )

    # manual_photos val: 1 clean + 1 blackout aug
    _add_s1_item(
        "val",
        "manual_val_clean",
        _make_crisp_tabletop_image(201),
        box_poly,
        "manual_photos",
        "manual/g_val",
    )
    _add_s1_item(
        "val",
        "manual_val_blackout",
        np.full((160, 240, 3), 5, dtype=np.uint8),
        box_poly,
        "manual_photos",
        "manual/g_val",
    )

    # manual_photos test: include a dark frame to prove `test` is NEVER filtered or altered!
    dark_test_img = np.full((160, 240, 3), 12, dtype=np.uint8)
    dark_test_img[20:40, 20:40] = 25
    _add_s1_item(
        "test",
        "manual_test_untouched",
        dark_test_img,
        slim_poly,
        "manual_photos",
        "manual/g_test",
    )

    # Non-target Stage-1 train replay candidates (hard negatives + positives)
    for idx in range(3):
        _add_s1_item(
            "train",
            f"oi_neg_pen_{idx}",
            _make_crisp_tabletop_image(300 + idx),
            "",
            "open_images_negatives",
            f"oi/pen_{idx}",
            negative=True,
        )
    _add_s1_item(
        "train",
        "kaggle_battery_neg",
        _make_crisp_tabletop_image(350),
        "",
        "kaggle_train",
        "kaggle/bat_0",
        negative=True,
        removed_battery=2,
    )
    for idx in range(4):
        _add_s1_item(
            "train",
            f"rf_primary_pos_{idx}",
            _make_crisp_tabletop_image(400 + idx),
            slim_poly,
            "roboflow_primary",
            f"rf/pos_{idx}",
        )
    # Non-target test item (must NEVER appear in Stage-2 train/val/test!)
    _add_s1_item(
        "test",
        "rf_primary_heldout_test",
        _make_crisp_tabletop_image(999),
        slim_poly,
        "roboflow_primary",
        "rf/test_0",
    )

    (s1_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_rows) + "\n",
        encoding="utf-8",
    )

    s2_root = tmp_path / "stage2_manual_clean"
    summary = build_stage2_tabletop_dataset(
        stage1_dataset_root=s1_root,
        output_stage2_root=s2_root,
        target_source="manual_photos",
        replay_ratio=2.0,
        filter_corrupted_augs=True,
        oversample_compact_factor=2,
        seed=42,
    )

    assert summary["yaml_path"] == s2_root / "data.yaml"
    assert summary["quality_train"].removed_count == 1
    assert summary["quality_val"].removed_count == 1
    assert summary["oversampled_stems"] == ["manual_box_clean"]
    # 2 clean manual_photos train -> replay_ratio=2.0 -> 4 replay samples (2 hard neg + 2 pos)
    assert summary["replay_count"] == 4
    # 2 clean manual_photos + 1 oversampled _dup1 + 4 replay = 7 train, 1 val, 1 test
    assert summary["split_counts"] == {"train": 7, "val": 1, "test": 1}

    train_names = {p.name for p in (s2_root / "images" / "train").glob("*.png")}
    val_names = {p.name for p in (s2_root / "images" / "val").glob("*.png")}
    test_names = {p.name for p in (s2_root / "images" / "test").glob("*.png")}

    assert "manual_box_clean.png" in train_names
    assert "manual_box_clean_dup1.png" in train_names
    assert "manual_box_whiteout.png" not in train_names
    assert val_names == {"manual_val_clean.png"}
    assert test_names == {"manual_test_untouched.png"}
    assert "rf_primary_heldout_test.png" not in (train_names | val_names | test_names)

    # Verify hard-negative replay was included
    assert any(
        name.startswith("oi_neg_pen_") or name == "kaggle_battery_neg.png" for name in train_names
    )

    # Verify zero cross-split SHA-256 overlap across train, val, and test
    split_hashes = {
        sp: {_sha256_file(p) for p in (s2_root / "images" / sp).glob("*.png")}
        for sp in ("train", "val", "test")
    }
    assert (split_hashes["train"] & split_hashes["val"]) == set()
    assert (split_hashes["train"] & split_hashes["test"]) == set()
    assert (split_hashes["val"] & split_hashes["test"]) == set()


def test_colab_train_cli_flags_and_stage2_unfrozen_training_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--drive-dir",
            str(tmp_path / "drive"),
            "--skip-stage1",
            "--run-stage2-manual",
            "--no-filter-corrupted-augs",
            "--stage2-replay-ratio",
            "1.5",
            "--stage2-imgsz",
            "960",
        ]
    )
    assert args.skip_stage1 is True
    assert args.run_stage2_manual is True
    assert args.filter_corrupted_augs is False
    assert args.stage2_replay_ratio == pytest.approx(1.5)
    assert args.stage2_imgsz == 960

    # Prepare a minimal Drive folder with pre-existing Stage-1 weights so Stage-1 is skipped
    drive_dir = tmp_path / "drive"
    vape_dir = drive_dir / "vape_yolo"
    (vape_dir / "images").mkdir(parents=True)
    (vape_dir / "labels").mkdir(parents=True)

    poly = "0 0.25 0.30 0.55 0.30 0.55 0.55 0.25 0.55\n"
    manifest_lines: list[str] = []
    for sp, idx in (("train", 0), ("val", 1), ("test", 2)):
        stem = f"m_{sp}_{idx}"
        cv2.imwrite(str(vape_dir / "images" / f"{stem}.png"), _make_crisp_tabletop_image(50 + idx))
        (vape_dir / "labels" / f"{stem}.txt").write_text(poly, encoding="utf-8")
        manifest_lines.append(
            json.dumps(
                {
                    "image": f"images/{stem}.png",
                    "source_id": "manual_photos",
                    "origin_group": f"m/{sp}_{idx}",
                    "negative": False,
                    "split": sp,
                }
            )
        )
    (vape_dir / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    s1_ckpt = drive_dir / "vape_stage1_yolo11s_seg_best.pt"
    s1_ckpt.write_bytes(b"fake-stage1-weights")

    train_calls: list[dict[str, object]] = []
    val_calls: list[dict[str, object]] = []

    class _FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights
            self.trainer: object | None = None

        def train(self, **kwargs: object) -> SimpleNamespace:
            train_calls.append({"weights": self.weights, **kwargs})
            save_dir = tmp_path / "fake_run"
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            best_file = save_dir / "weights" / "best.pt"
            best_file.write_bytes(b"fake-stage2-weights")
            self.trainer = SimpleNamespace(best=str(best_file))
            return SimpleNamespace(save_dir=str(save_dir))

        def val(self, **kwargs: object) -> SimpleNamespace:
            val_calls.append({"weights": self.weights, **kwargs})
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))
    monkeypatch.setattr(
        "leukerbad.dataset.colab_train._select_gpu_train_params",
        lambda: {"device": "cpu", "batch": 8, "workers": 2, "cache": False},
    )

    best_out = prepare_and_train_from_drive(
        drive_dir=drive_dir,
        work_dir=tmp_path / "work",
        skip_stage1=True,
        run_stage2_manual=True,
        stage2_imgsz=768,
    )
    assert best_out.is_file()
    assert (drive_dir / "vape_stage2_yolo11s_seg_best.pt").is_file()

    # Stage 1 was skipped; only Stage 2 trained, with freeze=None, lr0=3e-4, cos_lr=True, hsv_h=0.04
    assert len(train_calls) == 1
    s2_call = train_calls[0]
    assert s2_call["weights"] == str(s1_ckpt)
    assert s2_call["freeze"] is None
    assert s2_call["lr0"] == pytest.approx(3e-4)
    assert s2_call["cos_lr"] is True
    assert s2_call["hsv_h"] == pytest.approx(0.04)
    assert s2_call["imgsz"] == 768

    # Both Stage-2 test AND Stage-1 full held-out test were evaluated!
    assert len(val_calls) == 2
    eval_datasets = [str(c["data"]) for c in val_calls]
    assert any("stage2_manual_clean" in d for d in eval_datasets)
    assert any("vape_yolo_seg" in d for d in eval_datasets)


class _AreaGatedStubDetector:
    """Stub detector mimicking YOLO 640 downscale drop: only detects blobs >= min_rel_area."""

    def __init__(self, min_rel_area: float = 0.04, mm_per_px: float = 0.5) -> None:
        self.min_rel_area = min_rel_area
        self.mm_per_px = mm_per_px

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(gray.shape[0] * gray.shape[1])

        dets: list[Detection] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_rel_area * frame_area:
                continue
            poly = cnt.reshape(-1, 2).astype(np.float32)
            if poly.shape[0] < 3:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bbox = (int(x), int(y), int(x + w), int(y + h))
            grasp = compute_grasp_from_polygon(poly)
            meta: dict[str, object] = {"class_id": 0}
            if grasp is not None:
                w_mm, open_mm = estimate_gripper_opening_mm(grasp.width_px, self.mm_per_px)
                meta["width_mm"] = w_mm
                meta["gripper_opening_mm"] = open_mm
            dets.append(
                Detection(
                    label="vape",
                    confidence=0.88,
                    bounding_box=bbox,
                    polygon=poly,
                    grasp=grasp,
                    metadata=meta,
                )
            )
        return dets


def _draw_rotated_vape(
    canvas: np.ndarray,
    center: tuple[float, float],
    length_px: float,
    width_px: float,
    angle_deg: float,
    color: tuple[int, int, int] = (230, 240, 250),
) -> None:
    rect = (center, (length_px, width_px), angle_deg)
    box = np.round(cv2.boxPoints(rect)).astype(np.int32)
    cv2.fillPoly(canvas, [box], color)


def test_apply_clahe_bgr_and_tile_windows() -> None:
    low_contrast = np.full((120, 160, 3), 90, dtype=np.uint8)
    low_contrast[30:90, 40:120] = (115, 60, 180)
    enhanced = apply_clahe_bgr(low_contrast, clip_limit=2.5)
    assert enhanced.shape == low_contrast.shape
    assert enhanced.dtype == np.uint8
    assert float(enhanced.std()) > float(low_contrast.std())

    gray_2d = cv2.cvtColor(low_contrast, cv2.COLOR_BGR2GRAY)
    enhanced_2d = apply_clahe_bgr(gray_2d)
    assert enhanced_2d.shape == gray_2d.shape

    windows = generate_tile_windows(1080, 1920, grid=(2, 2), overlap=0.25)
    assert len(windows) == 4
    assert windows[0][0] == 0 and windows[0][1] == 0
    assert windows[-1][2] == 1920 and windows[-1][3] == 1080
    # Central seam (960, 540) must be generously inside all 4 overlapping tiles
    for x1, y1, x2, y2 in windows:
        assert x1 < 900 and x2 > 1020
        assert y1 < 500 and y2 > 580


def test_tiled_detector_recovers_small_1080p_vapes_and_merges_center_seam_duplicates() -> None:
    base_detector = _AreaGatedStubDetector(min_rel_area=0.04, mm_per_px=0.5)
    tiled_detector = TiledYoloSegDetector(base_detector, grid=(2, 2), overlap=0.25, use_clahe=True)

    # 1. Two small rotated vapes on 1080x1920 (~1.5% of full frame each, ~4.7% of each 2x2 tile)
    frame = np.full((1080, 1920, 3), 35, dtype=np.uint8)
    _draw_rotated_vape(frame, center=(420.0, 260.0), length_px=360.0, width_px=88.0, angle_deg=30.0)
    _draw_rotated_vape(
        frame, center=(1480.0, 810.0), length_px=360.0, width_px=88.0, angle_deg=-40.0
    )

    assert base_detector.detect(frame) == []

    recovered = tiled_detector.detect(frame)
    assert len(recovered) == 2

    by_x = sorted(recovered, key=lambda d: d.grasp.center_px[0] if d.grasp else 0.0)
    left_det, right_det = by_x[0], by_x[1]

    assert left_det.grasp is not None
    assert left_det.grasp.center_px[0] == pytest.approx(420.0, abs=4.0)
    assert left_det.grasp.center_px[1] == pytest.approx(260.0, abs=4.0)
    assert left_det.grasp.object_angle_deg == pytest.approx(30.0, abs=3.0)
    assert left_det.grasp.gripper_angle_deg == pytest.approx(-60.0, abs=3.0)
    assert left_det.metadata["width_mm"] == pytest.approx(44.0, abs=3.0)
    assert left_det.metadata["gripper_opening_mm"] == pytest.approx(56.0, abs=3.0)

    assert right_det.grasp is not None
    assert right_det.grasp.center_px[0] == pytest.approx(1480.0, abs=4.0)
    assert right_det.grasp.center_px[1] == pytest.approx(810.0, abs=4.0)
    assert right_det.grasp.object_angle_deg == pytest.approx(-40.0, abs=3.0)
    assert right_det.grasp.gripper_angle_deg == pytest.approx(50.0, abs=3.0)

    # 2. Single vape sitting squarely on the center (960, 540) 4-tile overlap seam -> merged into 1!
    seam_frame = np.full((1080, 1920, 3), 35, dtype=np.uint8)
    _draw_rotated_vape(
        seam_frame, center=(960.0, 540.0), length_px=220.0, width_px=145.0, angle_deg=0.0
    )
    assert base_detector.detect(seam_frame) == []

    seam_dets = tiled_detector.detect(seam_frame)
    assert len(seam_dets) == 1
    merged = seam_dets[0]
    assert merged.grasp is not None
    assert merged.grasp.center_px[0] == pytest.approx(960.0, abs=4.0)
    assert merged.grasp.center_px[1] == pytest.approx(540.0, abs=4.0)
    assert merged.metadata["tiled"]["merged_count"] == 4

    # 3. Seam-clipped overlapping halves merge into full vape length via polygon union
    class _SeamClippedStub:
        mm_per_px = 0.5

        def detect(self, sub_frame: np.ndarray) -> list[Detection]:
            if sub_frame.shape[1] > 1500:
                return []
            gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out: list[Detection] = []
            for cnt in cnts:
                if cv2.contourArea(cnt) < 5000.0:
                    continue
                poly = cnt.reshape(-1, 2).astype(np.float32)
                x, y, w, h = cv2.boundingRect(cnt)
                out.append(
                    Detection(
                        label="vape",
                        confidence=0.84,
                        bounding_box=(int(x), int(y), int(x + w), int(y + h)),
                        polygon=poly,
                        grasp=compute_grasp_from_polygon(poly),
                    )
                )
            return out

    long_seam_frame = np.full((1080, 1920, 3), 35, dtype=np.uint8)
    _draw_rotated_vape(
        long_seam_frame, center=(960.0, 260.0), length_px=420.0, width_px=76.0, angle_deg=0.0
    )
    seam_union_dets = TiledYoloSegDetector(_SeamClippedStub(), grid=(2, 2), overlap=0.25).detect(
        long_seam_frame
    )
    assert len(seam_union_dets) == 1
    assert seam_union_dets[0].grasp is not None
    assert seam_union_dets[0].grasp.length_px == pytest.approx(420.0, abs=6.0)
    assert seam_union_dets[0].grasp.center_px[0] == pytest.approx(960.0, abs=4.0)
