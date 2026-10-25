"""
Patient-level BraTS preprocessing for 2D MSUQ-Net training and 3D test evaluation.

Outputs:
  processed_data/trainImage, trainMask  : tumor-containing 2D slices from train patients
  processed_data/valImage, valMask      : tumor-containing 2D slices from validation patients
  processed_data/testImage, testMask    : ALL 2D slices from held-out test patients
  processed_data/testVolumes            : one NPZ per test patient with complete cropped 3D data
  processed_data/manifests/split.json   : exact patient-level split

The model input remains 160x160 with channels [FLAIR, T1, T1CE, T2].
Masks are saved as original BraTS labels (0, 1, 2, 4); dataset.py converts them to WT/TC/ET.

IMPORTANT:
- Check --brats_path and the suffixes below against your local files.
- Back up/remove the old processed_data output before running; stale files can contaminate splits.
- This script creates a NEW 60/20/20 patient split. Use this same split for training and testing.
- Update train3.py to load trainImage/trainMask and valImage/valMask directly; do NOT randomly split
  trainImage slice filenames again.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk


SUFFIXES = {
    "flair": "_flair.nii",
    "t1": "_t1.nii",
    "t1ce": "_t1ce.nii",
    "t2": "_t2.nii",
    "seg": "_seg.nii",
}


def resolve_nifti(patient_dir: Path, patient_id: str, suffix: str) -> Path:
    """Resolve .nii or .nii.gz files."""
    candidates = [
        patient_dir / f"{patient_id}{suffix}",
        patient_dir / f"{patient_id}{suffix}.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing file for patient {patient_id}, suffix {suffix}. "
        f"Checked: {[str(p) for p in candidates]}"
    )


def read_patient_image(patient_dir: Path, patient_id: str, suffix: str):
    path = resolve_nifti(patient_dir, patient_id, suffix)
    return sitk.ReadImage(str(path)), path


def normalize_volume(volume, upper_percentile=99, lower_percentile=1):
    """Match the original preprocessing style: clip and standardize nonzero voxels."""
    volume = volume.astype(np.float32, copy=False)
    upper = np.percentile(volume, upper_percentile)
    lower = np.percentile(volume, lower_percentile)
    clipped = np.clip(volume, lower, upper)
    nonzero = clipped[clipped != 0]

    if nonzero.size == 0 or np.std(clipped) == 0 or np.std(nonzero) == 0:
        return clipped.astype(np.float32)

    result = (clipped - float(np.mean(nonzero))) / float(np.std(nonzero))
    result[result == result.min()] = -9.0
    return result.astype(np.float32)


def center_crop(volume, crop_h=160, crop_w=160):
    """Input/output order is [Z, Y, X]."""
    _, height, width = volume.shape
    if crop_h > height or crop_w > width:
        raise ValueError(
            f"Cannot center-crop {crop_h}x{crop_w} from input {height}x{width}."
        )
    y0 = height // 2 - crop_h // 2
    x0 = width // 2 - crop_w // 2
    return volume[:, y0:y0 + crop_h, x0:x0 + crop_w], y0, x0


def same_geometry(reference, candidate, patient_id, name):
    if (
        reference.GetSize() != candidate.GetSize()
        or reference.GetSpacing() != candidate.GetSpacing()
        or reference.GetOrigin() != candidate.GetOrigin()
        or reference.GetDirection() != candidate.GetDirection()
    ):
        raise ValueError(
            f"{patient_id}: {name} geometry differs from FLAIR. "
            "Resample all modalities and the segmentation to one reference grid first."
        )


def load_and_process_patient(patient_dir, crop_h, crop_w):
    patient_id = patient_dir.name
    loaded = {
        key: read_patient_image(patient_dir, patient_id, suffix)[0]
        for key, suffix in SUFFIXES.items()
    }
    reference = loaded["flair"]
    for key in ("t1", "t1ce", "t2", "seg"):
        same_geometry(reference, loaded[key], patient_id, key)

    arrays = {
        key: sitk.GetArrayFromImage(image)
        for key, image in loaded.items()
    }
    mask_full = arrays["seg"].astype(np.uint8)

    # Keep the mask labels intact; dataset.py maps labels to WT/TC/ET.
    modalities = []
    for key in ("flair", "t1", "t1ce", "t2"):
        normalized = normalize_volume(arrays[key])
        cropped, y0, x0 = center_crop(normalized, crop_h, crop_w)
        modalities.append(cropped)

    mask_crop, y0_mask, x0_mask = center_crop(mask_full, crop_h, crop_w)
    if (y0, x0) != (y0_mask, x0_mask):
        raise RuntimeError("Inconsistent crop offsets.")

    # [Z, H, W, C], same channel order expected by the current Dataset class.
    image_4ch = np.stack(modalities, axis=-1).astype(np.float32)

    return {
        "patient_id": patient_id,
        "image": image_4ch,
        "mask_crop": mask_crop.astype(np.uint8),
        "mask_full": mask_full,
        "num_slices": int(mask_full.shape[0]),
        "spacing_xyz": np.asarray(reference.GetSpacing(), dtype=np.float32),
        "origin_xyz": np.asarray(reference.GetOrigin(), dtype=np.float64),
        "direction": np.asarray(reference.GetDirection(), dtype=np.float64),
        "original_size_xyz": np.asarray(reference.GetSize(), dtype=np.int32),
        "crop_y0": int(y0),
        "crop_x0": int(x0),
        "crop_h": int(crop_h),
        "crop_w": int(crop_w),
    }


def save_slices(patient_data, image_dir, mask_dir, tumor_only):
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = patient_data["image"]
    mask = patient_data["mask_crop"]
    patient_id = patient_data["patient_id"]
    saved = 0

    for z in range(mask.shape[0]):
        if tumor_only and not np.any(mask[z] != 0):
            continue
        # Keep the original z index, even when tumor-only slices are saved.
        np.save(image_dir / f"{patient_id}_{z}.npy", image[z])
        np.save(mask_dir / f"{patient_id}_{z}.npy", mask[z])
        saved += 1
    return saved


def split_patients(patient_ids, train_ratio, val_ratio, seed):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be >0 and sum to less than 1.")
    ids = list(patient_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if min(n_train, n_val, n - n_train - n_val) < 1:
        raise ValueError(f"Not enough patients ({n}) for requested three-way split.")
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brats_path",
        type=str,
        default=r"BraTS2021_Training_Data",
        help="Directory containing patient subfolders.",
    )
    parser.add_argument("--output_dir", type=str, default="./processed_data")
    parser.add_argument("--roi_height", type=int, default=160)
    parser.add_argument("--roi_width", type=int, default=160)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (args.roi_height, args.roi_width) != (160, 160):
        print(
            f"WARNING: configured ROI is {args.roi_height}x{args.roi_width}; "
            "confirm this matches the model/checkpoint configuration."
        )

    root = Path(args.brats_path)
    out = Path(args.output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"BraTS directory not found: {root.resolve()}")

    patient_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(patient_dirs) < 3:
        raise RuntimeError(f"Found only {len(patient_dirs)} patient directories in {root}")

    # Fail rather than silently mixing old slices from a previous run.
    expected_dirs = [
        out / "trainImage", out / "trainMask",
        out / "valImage", out / "valMask",
        out / "testImage", out / "testMask",
        out / "testVolumes",
    ]
    nonempty = [str(d) for d in expected_dirs if d.exists() and any(d.iterdir())]
    if nonempty:
        raise FileExistsError(
            "Output directories already contain files, so stale data could be mixed in:\n"
            + "\n".join(nonempty)
            + "\nBack up/remove these generated directories or choose a new --output_dir."
        )

    train_ids, val_ids, test_ids = split_patients(
        [p.name for p in patient_dirs],
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    patient_dir_by_id = {p.name: p for p in patient_dirs}

    for d in expected_dirs:
        d.mkdir(parents=True, exist_ok=True)
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1 - args.train_ratio - args.val_ratio, 6),
        "roi_height": args.roi_height,
        "roi_width": args.roi_width,
        "train_patients": train_ids,
        "validation_patients": val_ids,
        "test_patients": test_ids,
    }
    with open(manifest_dir / "patient_split.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Total patients: {len(patient_dirs)}")
    print(f"Train: {len(train_ids)} patients")
    print(f"Validation: {len(val_ids)} patients")
    print(f"Test: {len(test_ids)} patients")
    print(f"ROI: {args.roi_height} x {args.roi_width}")
    print("Split manifest saved to:", manifest_dir / "patient_split.json")

    # Training and validation: retain tumor-containing slices, matching original intent.
    for split_name, ids, image_dir, mask_dir in [
        ("TRAIN", train_ids, out / "trainImage", out / "trainMask"),
        ("VAL", val_ids, out / "valImage", out / "valMask"),
    ]:
        for i, patient_id in enumerate(ids, start=1):
            pdata = load_and_process_patient(
                patient_dir_by_id[patient_id], args.roi_height, args.roi_width
            )
            n_saved = save_slices(pdata, image_dir, mask_dir, tumor_only=True)
            print(f"[{split_name} {i}/{len(ids)}] {patient_id}: {n_saved} tumor slices")

    # Test: save every slice plus a patient-level volume file for reconstruction/evaluation.
    for i, patient_id in enumerate(test_ids, start=1):
        pdata = load_and_process_patient(
            patient_dir_by_id[patient_id], args.roi_height, args.roi_width
        )
        n_saved = save_slices(
            pdata, out / "testImage", out / "testMask", tumor_only=False
        )
        volume_file = out / "testVolumes" / f"{patient_id}.npz"
        np.savez_compressed(
            volume_file,
            patient_id=np.asarray(patient_id),
            image=pdata["image"],                 # [Z, 160, 160, 4]
            mask_crop=pdata["mask_crop"],         # [Z, 160, 160], labels 0/1/2/4
            mask_full=pdata["mask_full"],         # original full-size label volume
            num_slices=np.asarray(pdata["num_slices"], dtype=np.int32),
            spacing_xyz=pdata["spacing_xyz"],
            origin_xyz=pdata["origin_xyz"],
            direction=pdata["direction"],
            original_size_xyz=pdata["original_size_xyz"],
            crop_y0=np.asarray(pdata["crop_y0"], dtype=np.int32),
            crop_x0=np.asarray(pdata["crop_x0"], dtype=np.int32),
            crop_h=np.asarray(pdata["crop_h"], dtype=np.int32),
            crop_w=np.asarray(pdata["crop_w"], dtype=np.int32),
        )
        print(
            f"[TEST {i}/{len(test_ids)}] {patient_id}: saved {n_saved} slices; "
            f"volume file {volume_file.name}"
        )

    print("\nPreprocessing completed.")
    print("IMPORTANT: update train3.py to load trainImage/trainMask and valImage/valMask")
    print("directly. Do not run a second random split over slice filenames.")
    print("Use testVolumes and the manifest for held-out patient-level evaluation.")


if __name__ == "__main__":
    main()
