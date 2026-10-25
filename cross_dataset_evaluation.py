#!/usr/bin/env python3
"""BayesianEMCCUncertaintyModel (models/model.py) evaluation with patient-level 3D reconstruction.

Uses the prepared dataset produced by prepare_brats_patient_split_2d_and_3d.py:
  processed_data/testVolumes/<patient_id>.npz
  processed_data/manifests/patient_split.json

The Bayesian EMCC network remains 2D. It predicts every axial slice; predictions are restored to
full original in-plane dimensions and evaluated as 3D patient volumes.

Outputs:
  per_patient_3d_metrics.csv
  summary_3d_metrics.csv
  predictions/<patient_id>_predictions.npz
  predictions/<patient_id>_{WT,TC,ET}.nii.gz (if SimpleITK is installed)

Important: use the checkpoint produced by the same BayesianEMCCUncertaintyModel architecture and
preprocessing pipeline. Thresholds must be fixed before test evaluation; do not tune
thresholds on the held-out test patients.
"""

import argparse
import csv
import json
import os
import sys
import importlib
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage as ndi

from models.model import BayesianEMCCUncertaintyModel

warnings.filterwarnings("ignore")
REGIONS = ("WT", "TC", "ET")


def parse_args():
    p = argparse.ArgumentParser(description="Bayesian EMCC patient-level 3D evaluation")
    p.add_argument("--test_volumes_dir", default="processed_data/testVolumes")
    p.add_argument("--manifest_path", default="processed_data/manifests/patient_split.json")
    p.add_argument("--model_path", default="checkpoint_bayesian_uncertainty_EMCC/2021/best_model.pth")
    p.add_argument("--output_dir", default="bayesian_EMCC_3d_brats2020_reverse_overlap_excluded")
    p.add_argument("--exclusion_file", default="brats2021_to_brats2020_reverse_overlap_audit/brats2020_exclusion_candidates.txt", help="Reverse-audit strong-candidate BraTS 2020 test IDs, one per line")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--drop_rate", type=float, default=0.20, help="Match Bayesian EMCC training script")
    p.add_argument("--mc_samples", type=int, default=1,
                   help="Bayesian weight-sampling passes per slice; ordinary dropout and BatchNorm remain in eval mode.")
    p.add_argument("--allow_nonmatching_training_split", action="store_true",
                   help="Not recommended: allow evaluating volumes despite missing proof of training split identity.")
    p.add_argument("--expected_test_count", type=int, default=None,
                   help="Optional assertion for BraTS 2020 test cohort size (e.g., 75 for a specific split)")
    p.add_argument("--expected_exclusion_count", type=int, default=None,
                   help="Optional assertion for strong-candidate count (e.g., 56 for a specific audit)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_nifti", action="store_true", help="Save each predicted region as NIfTI (requires SimpleITK)")
    return p.parse_args()


def get_patient_id(path):
    return path.name[:-4] if path.name.endswith(".npz") else path.stem


def load_manifest_test_ids(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    ids = manifest.get("test_patients")
    if not isinstance(ids, list) or not ids:
        raise ValueError("Manifest has no non-empty 'test_patients' list. Check the preprocessing output.")
    return set(str(x) for x in ids)



def load_strong_candidate_exclusions(exclusion_file):
    path = Path(exclusion_file)
    if not path.is_file():
        raise FileNotFoundError(f"Reverse-audit exclusion list not found: {path}")
    lines = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not lines:
        raise ValueError("Reverse-audit exclusion list is empty; cannot run overlap-excluded analysis")
    if len(lines) != len(set(lines)):
        raise ValueError("Duplicate patient IDs in reverse-audit exclusion list")
    for patient_id in lines:
        if not patient_id.startswith("BraTS20_Training_"):
            raise ValueError(f"Unexpected BraTS 2020 patient ID: {patient_id}")
    return set(lines)


def load_model(checkpoint_path, device, drop_rate):
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = BayesianEMCCUncertaintyModel(out_channels=3, drop_rate=drop_rate)
    # Load only checkpoints that you produced or otherwise trust.
    # Compatibility shim: checkpoints serialized with NumPy 2.x may reference
    # ``numpy._core``; older NumPy releases expose the same modules as
    # ``numpy.core``. Alias these module paths before unpickling. This does not
    # bypass checkpoint safety; only load a checkpoint from a trusted source.
    try:
        import numpy._core  # noqa: F401  (available in NumPy 2.x)
    except ModuleNotFoundError:
        try:
            import numpy.core as _np_core
            sys.modules.setdefault("numpy._core", _np_core)
            for _submodule in (
                "multiarray", "numeric", "_multiarray_umath", "umath",
                "overrides", "_dtype", "numerictypes", "_methods",
            ):
                try:
                    _module = importlib.import_module(f"numpy.core.{_submodule}")
                    sys.modules.setdefault(f"numpy._core.{_submodule}", _module)
                except ImportError:
                    pass
        except Exception as _compat_error:
            raise RuntimeError(
                "Could not establish NumPy pickle compatibility. Check your NumPy installation."
            ) from _compat_error
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
    else:
        state = checkpoint
        epoch = "unknown"
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"Loaded Bayesian EMCC checkpoint: {checkpoint_path} (epoch={epoch})")
    return model


def validate_disjoint_splits(manifest):
    names = {"train": "train_patients", "validation": "validation_patients", "test": "test_patients"}
    groups = {}
    for short, key in names.items():
        ids = manifest.get(key)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Manifest missing nonempty {key}")
        if len(ids) != len(set(map(str, ids))):
            raise ValueError(f"Duplicate patient IDs in {key}")
        groups[short] = set(map(str, ids))
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise RuntimeError(f"Patient overlap between {a} and {b}: {sorted(overlap)[:20]}")
    print("Split audit: no overlapping IDs across train / validation / test.")
    return groups


def check_checkpoint_split(args, manifest):
    # The manifest proves internally disjoint splits, but alone cannot prove which
    # split was used when producing the existing checkpoint.
    print("NOTE: A disjoint manifest does not establish that this checkpoint was trained on that split.")
    print("Verify your checkpoint's training patient IDs against this manifest before manuscript claims.")


def region_masks_from_labels(labels):
    """BraTS labels: 1=non-enhancing/necrotic core, 2=edema, 4=enhancing tumor."""
    labels = np.asarray(labels)
    return {
        "WT": np.isin(labels, (1, 2, 4)),
        "TC": np.isin(labels, (1, 4)),
        "ET": labels == 4,
    }


def dice_3d(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    denom = int(pred.sum()) + int(gt.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def surface_distances_mm(pred, gt, spacing_zyx):
    """Return (HD95, ASSD) in mm; NaN when exactly one mask is empty."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return float("nan"), float("nan")

    structure = ndi.generate_binary_structure(3, 1)
    pred_surface = pred ^ ndi.binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = gt ^ ndi.binary_erosion(gt, structure=structure, border_value=0)
    # Distance transforms are sampled in NumPy array order: (Z,Y,X).
    dt_to_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing_zyx)
    dt_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing_zyx)
    d_pred_to_gt = dt_to_gt[pred_surface]
    d_gt_to_pred = dt_to_pred[gt_surface]
    both = np.concatenate((d_pred_to_gt, d_gt_to_pred))
    hd95 = float(np.percentile(both, 95))
    assd = float((d_pred_to_gt.sum() + d_gt_to_pred.sum()) / (len(d_pred_to_gt) + len(d_gt_to_pred)))
    return hd95, assd


def infer_patient(model, image_zyxc, device, batch_size, mc_samples=1):
    """Input [Z,H,W,4], output mean MC probabilities [Z,3,H,W]."""
    if image_zyxc.ndim != 4 or image_zyxc.shape[-1] != 4:
        raise ValueError(f"Expected [Z,H,W,4], got {image_zyxc.shape}")
    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1")
    model.eval()
    all_batches = []
    with torch.inference_mode():
        for start in range(0, len(image_zyxc), batch_size):
            batch_np = image_zyxc[start:start + batch_size]
            batch = torch.from_numpy(np.ascontiguousarray(batch_np.transpose(0, 3, 1, 2))).float().to(device)
            cumulative = None
            for _ in range(mc_samples):
                # BayesianConv2d samples weights in eval mode; keep ordinary dropout disabled.
                output = model(batch)
                if not isinstance(output, dict) or "final_mean" not in output:
                    raise RuntimeError("BayesianEMCCUncertaintyModel expected dict with 'final_mean'.")
                prob = torch.sigmoid(output["final_mean"]).float()
                cumulative = prob if cumulative is None else cumulative + prob
            all_batches.append((cumulative / mc_samples).cpu().numpy())
    model.eval()
    return np.concatenate(all_batches, axis=0)


def embed_crop_to_full(crop_pred_zhwc, full_shape_zyx, y0, x0):
    """Place cropped prediction [Z,Hc,Wc,C] back into original [Z,Y,X,C] grid."""
    z, h, w = map(int, full_shape_zyx)
    crop_z, crop_h, crop_w, channels = crop_pred_zhwc.shape
    if crop_z != z:
        raise ValueError(f"Slice count mismatch: prediction Z={crop_z}, reference Z={z}")
    if y0 < 0 or x0 < 0 or y0 + crop_h > h or x0 + crop_w > w:
        raise ValueError(
            f"Crop offsets {(y0, x0)} with size {(crop_h, crop_w)} do not fit full shape {(z, h, w)}"
        )
    full = np.zeros((z, h, w, channels), dtype=np.uint8)
    full[:, y0:y0 + crop_h, x0:x0 + crop_w, :] = crop_pred_zhwc.astype(np.uint8)
    return full


def save_nifti_regions(pred_zyxc, npz, patient_id, out_dir):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("--save_nifti requires SimpleITK. Install it with: pip install SimpleITK") from exc
    spacing = tuple(float(v) for v in npz["spacing_xyz"])
    origin = tuple(float(v) for v in npz["origin_xyz"])
    direction = tuple(float(v) for v in npz["direction"])
    for i, region in enumerate(REGIONS):
        image = sitk.GetImageFromArray(pred_zyxc[..., i].astype(np.uint8))
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image.SetDirection(direction)
        sitk.WriteImage(image, str(out_dir / f"{patient_id}_{region}.nii.gz"))


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.mc_samples < 1:
        raise ValueError("mc_samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    volume_dir = Path(args.test_volumes_dir)
    if not volume_dir.is_dir():
        raise FileNotFoundError(f"Test volumes directory not found: {volume_dir}")
    volume_files = sorted(volume_dir.glob("*.npz"))
    if not volume_files:
        raise RuntimeError(f"No patient .npz volumes found in {volume_dir}")

    with open(args.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    splits = validate_disjoint_splits(manifest)
    check_checkpoint_split(args, manifest)
    manifest_test_ids = splits["test"]
    file_ids = {get_patient_id(p) for p in volume_files}
    missing = manifest_test_ids - file_ids
    unexpected = file_ids - manifest_test_ids
    if missing or unexpected:
        raise RuntimeError(
            "Test-volume files do not match manifest test_patients. "
            f"Missing files for IDs (first 10): {sorted(missing)[:10]}; "
            f"unexpected IDs (first 10): {sorted(unexpected)[:10]}. "
            "Check that preprocessing and checkpoint use the same split."
        )

    excluded_ids = load_strong_candidate_exclusions(args.exclusion_file)
    if not excluded_ids.issubset(manifest_test_ids):
        raise RuntimeError(
            "Candidate IDs are not contained in the BraTS 2020 test manifest: "
            f"{sorted(excluded_ids - manifest_test_ids)[:20]}"
        )
    if args.expected_test_count is not None and len(manifest_test_ids) != args.expected_test_count:
        raise RuntimeError(f"Expected {args.expected_test_count} BraTS 2020 test patients, found {len(manifest_test_ids)}")
    if args.expected_exclusion_count is not None and len(excluded_ids) != args.expected_exclusion_count:
        raise RuntimeError(f"Expected {args.expected_exclusion_count} exclusions, found {len(excluded_ids)}")
    included_ids = manifest_test_ids - excluded_ids
    if not included_ids:
        raise RuntimeError("No patients remain after candidate exclusion")
    print(f"BraTS 2020 test cohort: {len(manifest_test_ids)}; "
          f"excluded strong candidates: {len(excluded_ids)}; "
          f"retained (NOT proven independent): {len(included_ids)}")

    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_path, device, args.drop_rate)

    rows = []
    audit_rows = []
    for volume_path in tqdm(volume_files, desc="3D patient evaluation"):
        patient_id = get_patient_id(volume_path)
        if patient_id in excluded_ids:
            audit_rows.append({"patient_id": patient_id, "decision": "EXCLUDED_STRONG_CANDIDATE",
                               "reason": "Conservative exclusion after reverse BraTS 2021 train/validation vs BraTS 2020 test audit"})
            continue
        audit_rows.append({"patient_id": patient_id, "decision": "EVALUATED",
                           "reason": "No strong match detected; independence NOT established"})
        with np.load(volume_path, allow_pickle=False) as data:
            required = ["image", "mask_full", "spacing_xyz", "origin_xyz", "direction",
                        "original_size_xyz", "crop_y0", "crop_x0", "crop_h", "crop_w"]
            absent = [key for key in required if key not in data]
            if absent:
                raise KeyError(f"{volume_path} lacks required fields: {absent}")
            image = data["image"].astype(np.float32)  # [Z,160,160,4]
            if not np.isfinite(image).all():
                raise ValueError(f"{patient_id}: nonfinite image voxels")
            gt_labels_full = data["mask_full"].astype(np.uint8)  # [Z,Y,X]
            spacing_xyz = data["spacing_xyz"].astype(float)
            y0, x0 = int(data["crop_y0"]), int(data["crop_x0"])
            orig_size_xyz = data["original_size_xyz"].astype(int)

        expected_zyx = (int(orig_size_xyz[2]), int(orig_size_xyz[1]), int(orig_size_xyz[0]))
        if tuple(gt_labels_full.shape) != expected_zyx:
            raise ValueError(
                f"{patient_id}: mask_full shape {gt_labels_full.shape} does not match "
                f"original_size_xyz {tuple(orig_size_xyz)} (expected ZYX={expected_zyx})."
            )
        if image.ndim != 4 or image.shape[-1] != 4:
            raise ValueError(f"{patient_id}: expected [Z,H,W,4] image, got {image.shape}")
        if image.shape[1:3] != (160, 160):
            raise ValueError(f"{patient_id}: model expected 160x160 ROI, got {image.shape[1:3]}")
        if image.shape[0] != gt_labels_full.shape[0]:
            raise ValueError(f"{patient_id}: input slice count and reference volume differ.")

        probabilities = infer_patient(model, image, device, args.batch_size, args.mc_samples)  # [Z,3,Hc,Wc]
        # Convert to [Z,Hc,Wc,3], threshold using a fixed pre-specified threshold.
        pred_crop = (probabilities.transpose(0, 2, 3, 1) > args.threshold).astype(np.uint8)
        pred_full = embed_crop_to_full(
            pred_crop,
            gt_labels_full.shape,
            y0,
            x0,
        )
        gt_regions = region_masks_from_labels(gt_labels_full)

        row = {"patient_id": patient_id, "num_slices": int(image.shape[0])}
        for idx, region in enumerate(REGIONS):
            pred_region = pred_full[..., idx].astype(bool)
            gt_region = gt_regions[region]
            dice = dice_3d(pred_region, gt_region)
            hd95, assd = surface_distances_mm(
                pred_region, gt_region,
                spacing_zyx=(float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
            )
            row[f"{region}_Dice"] = dice
            row[f"{region}_HD95_mm"] = hd95
            row[f"{region}_ASSD_mm"] = assd

        np.savez_compressed(
            pred_dir / f"{patient_id}_predictions.npz",
            patient_id=np.asarray(patient_id),
            prediction_regions=pred_full,
            ground_truth_labels=gt_labels_full,
            threshold=np.asarray(args.threshold, dtype=np.float32),
            spacing_xyz=spacing_xyz,
        )
        if args.save_nifti:
            # Re-open NPZ to pass SimpleITK geometry metadata in the expected format.
            with np.load(volume_path, allow_pickle=False) as data:
                save_nifti_regions(pred_full, data, patient_id, pred_dir)
        rows.append(row)

    if len(rows) != len(included_ids):
        raise RuntimeError(f"Expected {len(included_ids)} included patients, evaluated {len(rows)}")
    with (out_dir / "exclusion_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "decision", "reason"])
        writer.writeheader()
        writer.writerows(audit_rows)
    with (out_dir / "excluded_candidate_patient_ids.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(sorted(excluded_ids)) + "\n")

    per_patient_csv = out_dir / "per_patient_3d_metrics.csv"
    fields = ["patient_id", "num_slices"] + [
        f"{region}_{metric}" for region in REGIONS
        for metric in ("Dice", "HD95_mm", "ASSD_mm")
    ]
    with open(per_patient_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for region in REGIONS:
        for metric in ("Dice", "HD95_mm", "ASSD_mm"):
            values = np.asarray([r[f"{region}_{metric}"] for r in rows], dtype=float)
            valid = values[np.isfinite(values)]
            summary_rows.append({
                "region": region,
                "metric": metric,
                "mean": float(np.mean(valid)) if valid.size else float("nan"),
                "std": float(np.std(valid, ddof=1)) if valid.size > 1 else (0.0 if valid.size == 1 else float("nan")),
                "n_patients_total": len(rows),
                "n_patients_included": int(valid.size),
                "n_undefined": int((~np.isfinite(values)).sum()),
            })

    summary_csv = out_dir / "summary_3d_metrics.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = {
        "model_architecture": "BayesianEMCCUncertaintyModel models/model.py",
        "mc_samples": args.mc_samples,
        "sampling_policy": "BayesianConv2d weight sampling; ordinary dropout and BatchNorm disabled by eval mode",
        "drop_rate": args.drop_rate,
        "split_patient_counts": {name: len(pids) for name, pids in splits.items()},
        "checkpoint_split_provenance_verified": False,
        "experiment_direction": "BraTS 2021-trained model -> BraTS 2020 test",
        "external_validation_claim": "Related-release cross-dataset transfer only; screening does not establish independence",
        "model_path": str(Path(args.model_path)),
        "test_volumes_dir": str(volume_dir),
        "manifest_path": str(Path(args.manifest_path)),
        "threshold_per_region": {region: args.threshold for region in REGIONS},
        "number_of_test_patients": len(rows),
        "initial_brats2020_test_patients": len(manifest_test_ids),
        "strong_candidate_patients_conservatively_excluded": len(excluded_ids),
        "retained_patient_count_not_proven_independent": len(included_ids),
        "exclusion_source_file": str(args.exclusion_file),
        "expected_test_count_assertion": args.expected_test_count,
        "expected_exclusion_count_assertion": args.expected_exclusion_count,
        "exclusion_method": "Conservative sensitivity analysis: exclude all reverse-audit strong-candidate BraTS 2020 test IDs, without asserting confirmed same-patient identity",
        "distance_unit": "mm",
        "empty_mask_policy": {
            "both_empty": "Dice=1, HD95=0, ASSD=0",
            "exactly_one_empty": "Dice computed normally; HD95/ASSD undefined (NaN) and excluded from distance summary"
        },
        "metric_note": "3D metrics computed on full original grid; cropped predictions are zero-padded outside the 160x160 ROI.",
        "summary": summary_rows,
    }
    with open(out_dir / "summary_3d_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("\n3D patient-level evaluation completed.")
    print(f"Patients evaluated: {len(rows)}")
    print(f"Per-patient metrics: {per_patient_csv}")
    print(f"Summary metrics: {summary_csv}")
    print(f"Prediction volumes: {pred_dir}")
    print("\nSummary (mean ± SD across defined patient metrics):")
    for r in summary_rows:
        print(f"{r['region']:>2} {r['metric']:<10}: {r['mean']:.4f} ± {r['std']:.4f} "
              f"(n={r['n_patients_included']}/{r['n_patients_total']})")


if __name__ == "__main__":
    main()
