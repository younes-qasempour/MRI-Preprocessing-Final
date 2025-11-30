import os
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import SimpleITK as sitk
import argparse
from typing import List, Tuple, Dict, Any


def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkBSpline,
    is_label: bool = False,
) -> sitk.Image:
    """
    Resample a SimpleITK image to given isotropic spacing.

    - For intensity images (MRI), use BSpline interpolation.
    - For segmentation masks, use NearestNeighbor.
    - Preserve origin and direction.
    - Compute new size so that physical extent is preserved.
    - default_pixel_value:
        * For intensity images: use image minimum.
        * For masks: use 0.
    """

    # Normalize spacing as tuple of floats
    target_spacing = tuple(float(s) for s in target_spacing)

    original_spacing = np.array(image.GetSpacing(), dtype=float)
    original_size = np.array(image.GetSize(), dtype=int)

    # Compute physical extent (in mm) and derive the new size to preserve it
    extent = original_spacing * original_size.astype(float)
    new_size = np.maximum(1, np.round(extent / np.array(target_spacing, dtype=float)).astype(int))
    new_size = [int(x) for x in new_size]

    # Default pixel value
    if is_label:
        default_pixel_value = 0
        interpolator = sitk.sitkNearestNeighbor
    else:
        # Use minimum intensity as background for continuous images
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        default_pixel_value = float(stats.GetMinimum())

    # Configure resampler
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetDefaultPixelValue(default_pixel_value)

    resampled = resample_filter.Execute(image)

    # Ensure label images remain integer type
    if is_label:
        # Choose a safe integer type (UInt16 covers typical label ranges)
        if resampled.GetPixelID() not in (sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkUInt32, sitk.sitkInt16, sitk.sitkInt8, sitk.sitkInt32):
            resampled = sitk.Cast(resampled, sitk.sitkUInt16)

    return resampled


def validate_resampling(
    original_image: sitk.Image,
    resampled_image: sitk.Image,
    filename: str,
    patient_id: str,
    is_label: bool,
) -> Dict[str, Any]:
    """
    Compare original and resampled images and return a dict of validation metrics.
    """
    tol_iso = 1e-3
    tol_phys = 1.0  # mm per dimension
    tol_meta = 1e-3

    original_spacing = tuple(float(s) for s in original_image.GetSpacing())
    resampled_spacing = tuple(float(s) for s in resampled_image.GetSpacing())

    original_size = tuple(int(s) for s in original_image.GetSize())
    resampled_size = tuple(int(s) for s in resampled_image.GetSize())

    original_origin = tuple(float(x) for x in original_image.GetOrigin())
    resampled_origin = tuple(float(x) for x in resampled_image.GetOrigin())

    original_direction = tuple(float(x) for x in original_image.GetDirection())
    resampled_direction = tuple(float(x) for x in resampled_image.GetDirection())

    original_physical_size = (np.array(original_size) * np.array(original_spacing)).astype(float)
    resampled_physical_size = (np.array(resampled_size) * np.array(resampled_spacing)).astype(float)

    # Basic flags
    is_isotropic = all(abs(s - 1.0) < tol_iso for s in resampled_spacing)
    physical_dims_preserved = all(abs(o - r) <= tol_phys for o, r in zip(original_physical_size, resampled_physical_size))
    origin_preserved = all(abs(o - r) < tol_meta for o, r in zip(original_origin, resampled_origin))
    direction_preserved = all(abs(o - r) < tol_meta for o, r in zip(original_direction, resampled_direction))

    result: Dict[str, Any] = {
        "patient_id": patient_id,
        "filename": filename,
        "is_label": bool(is_label),
        "original_spacing": list(original_spacing),
        "resampled_spacing": list(resampled_spacing),
        "is_isotropic": bool(is_isotropic),
        "original_size": list(original_size),
        "resampled_size": list(resampled_size),
        "original_physical_size": list(original_physical_size),
        "resampled_physical_size": list(resampled_physical_size),
        "physical_dims_preserved": bool(physical_dims_preserved),
        "origin_preserved": bool(origin_preserved),
        "direction_preserved": bool(direction_preserved),
    }

    if not is_label:
        # Intensity stats
        o_stats = sitk.StatisticsImageFilter()
        o_stats.Execute(original_image)
        r_stats = sitk.StatisticsImageFilter()
        r_stats.Execute(resampled_image)
        result.update(
            {
                "original_min": float(o_stats.GetMinimum()),
                "original_max": float(o_stats.GetMaximum()),
                "original_mean": float(o_stats.GetMean()),
                "original_variance": float(o_stats.GetVariance()),
                "resampled_min": float(r_stats.GetMinimum()),
                "resampled_max": float(r_stats.GetMaximum()),
                "resampled_mean": float(r_stats.GetMean()),
                "resampled_variance": float(r_stats.GetVariance()),
            }
        )
    else:
        # Label set preservation
        orig_np = sitk.GetArrayViewFromImage(original_image)
        res_np = sitk.GetArrayViewFromImage(resampled_image)
        orig_labels = np.unique(orig_np).astype(int).tolist()
        res_labels = np.unique(res_np).astype(int).tolist()
        result.update(
            {
                "original_unique_labels": orig_labels,
                "resampled_unique_labels": res_labels,
                "labels_preserved": set(orig_labels) == set(res_labels),
            }
        )

    return result


def process_patient_images(
    images_root: Path,
    output_images_root: Path,
    target_spacing: Tuple[float, float, float],
    skip_existing: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    """Process all intensity MRI images under Images/PatientXX directories.

    Returns a tuple of (results_list, num_processed_files).
    """
    results: List[Dict[str, Any]] = []
    processed_count = 0

    patient_dirs = sorted([p for p in images_root.glob("Patient*") if p.is_dir()])
    for patient_dir in patient_dirs:
        out_patient_dir = output_images_root / patient_dir.name
        out_patient_dir.mkdir(parents=True, exist_ok=True)

        nii_paths = sorted(patient_dir.glob("*.nii.gz"))
        for nii_path in nii_paths:
            try:
                print(f"[{patient_dir.name}] Processing image {nii_path.name}")
                out_path = out_patient_dir / f"resampled_1mm_{nii_path.name}"
                if skip_existing and out_path.exists():
                    print(f"[{patient_dir.name}] Skipping existing {out_path.name}")
                    continue

                image = sitk.ReadImage(str(nii_path))
                resampled = resample_to_isotropic(
                    image,
                    target_spacing=target_spacing,
                    interpolator=sitk.sitkBSpline,
                    is_label=False,
                )

                sitk.WriteImage(resampled, str(out_path))
                print(f"[{patient_dir.name}] Saved resampled image to {out_path}")

                res = validate_resampling(
                    original_image=image,
                    resampled_image=resampled,
                    filename=nii_path.name,
                    patient_id=patient_dir.name,
                    is_label=False,
                )
                results.append(res)
                processed_count += 1
            except Exception as e:
                err = {
                    "patient_id": patient_dir.name,
                    "filename": nii_path.name,
                    "is_label": False,
                    "error_message": str(e),
                }
                print(f"[{patient_dir.name}] ERROR processing {nii_path.name}: {e}")
                results.append(err)

    return results, processed_count


def process_patient_masks(
    masks_root: Path,
    output_masks_root: Path,
    images_root: Path,
    target_spacing: Tuple[float, float, float],
    skip_existing: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    """Process segmentation masks corresponding to each PatientXX under Images/.

    Returns a tuple of (results_list, num_processed_masks).
    """
    results: List[Dict[str, Any]] = []
    processed_count = 0

    output_masks_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in images_root.glob("Patient*") if p.is_dir()])
    for patient_dir in patient_dirs:
        try:
            patient_id_str = patient_dir.name.replace("Patient", "")
            patient_id_int = int(patient_id_str)
            mask_name = f"GBM{patient_id_int:03d}.nii.gz.nii.seg.nrrd"
            mask_path = masks_root / mask_name

            if not mask_path.exists():
                print(f"[GBM{patient_id_int:03d}]    WARNING: Missing segmentation {mask_name}")
                results.append(
                    {
                        "patient_id": patient_dir.name,
                        "filename": mask_name,
                        "is_label": True,
                        "error_message": "mask_not_found",
                    }
                )
                continue

            out_mask_path = output_masks_root / f"resampled_1mm_{mask_name}"
            print(f"[GBM{patient_id_int:03d}]    Processing segmentation {mask_name}")
            if skip_existing and out_mask_path.exists():
                print(f"[GBM{patient_id_int:03d}]    Skipping existing {out_mask_path.name}")
                continue

            mask_img = sitk.ReadImage(str(mask_path))
            resampled_mask = resample_to_isotropic(
                mask_img,
                target_spacing=target_spacing,
                interpolator=sitk.sitkNearestNeighbor,
                is_label=True,
            )

            # Ensure integer label type
            if resampled_mask.GetPixelID() not in (sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkUInt32, sitk.sitkInt16, sitk.sitkInt8, sitk.sitkInt32):
                resampled_mask = sitk.Cast(resampled_mask, sitk.sitkUInt16)

            sitk.WriteImage(resampled_mask, str(out_mask_path))
            print(f"[GBM{patient_id_int:03d}]    Saved resampled segmentation to {out_mask_path}")

            res = validate_resampling(
                original_image=mask_img,
                resampled_image=resampled_mask,
                filename=mask_name,
                patient_id=patient_dir.name,
                is_label=True,
            )
            results.append(res)
            processed_count += 1
        except Exception as e:
            err = {
                "patient_id": patient_dir.name,
                "filename": mask_name if 'mask_name' in locals() else "<unknown>",
                "is_label": True,
                "error_message": str(e),
            }
            gbm_tag = f"GBM{patient_id_int:03d}" if 'patient_id_int' in locals() else "GBM???"
            print(f"[{gbm_tag}]    ERROR processing segmentation: {e}")
            results.append(err)

    return results, processed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample MRI images and segmentations to 1x1x1 mm isotropic spacing.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/younes/PycharmProjects/MRI-Preprocessing/30-nov-Final-MRI-Data",
        help="Root directory containing Images/ and Masks/",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/younes/PycharmProjects/MRI-Preprocessing/30-nov-Final-MRI-Data-1mm",
        help="Output root directory where resampled data will be written (mirrors structure).",
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Target isotropic spacing as three floats: sx sy sz",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose resampled outputs already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = Path(args.root_dir)
    images_root = root_dir / "Images"
    masks_root = root_dir / "Masks"

    output_root = Path(args.output_root)
    output_images_root = output_root / "Images"
    output_masks_root = output_root / "Masks"

    # Ensure output directories exist
    output_images_root.mkdir(parents=True, exist_ok=True)
    output_masks_root.mkdir(parents=True, exist_ok=True)

    target_spacing = tuple(args.target_spacing)
    skip_existing = bool(args.skip_existing)

    all_results: List[Dict[str, Any]] = []

    print("Starting resampling of intensity images...")
    image_results, n_images = process_patient_images(
        images_root=images_root,
        output_images_root=output_images_root,
        target_spacing=target_spacing,
        skip_existing=skip_existing,
    )
    all_results.extend(image_results)

    print("Starting resampling of segmentation masks...")
    mask_results, n_masks = process_patient_masks(
        masks_root=masks_root,
        output_masks_root=output_masks_root,
        images_root=images_root,
        target_spacing=target_spacing,
        skip_existing=skip_existing,
    )
    all_results.extend(mask_results)

    # Create DataFrame and save CSV
    df = pd.DataFrame(all_results)

    # Print table-like summary to stdout
    summary_cols = [
        "patient_id",
        "filename",
        "is_label",
        "is_isotropic",
        "physical_dims_preserved",
        "origin_preserved",
        "direction_preserved",
        "labels_preserved",
        "error_message",
    ]
    # Ensure all expected columns exist
    for col in summary_cols:
        if col not in df.columns:
            df[col] = None

    print("\nSummary (first 100 rows):")
    print(df[summary_cols].head(100).to_string(index=False))

    # Overall flags (ignore rows with errors/NaNs where applicable)
    def all_true(series: pd.Series) -> bool:
        if series.dropna().empty:
            return True
        return bool(series.dropna().astype(bool).all())

    all_isotropic = all_true(df["is_isotropic"]) if "is_isotropic" in df else True
    all_dims_preserved = all_true(df["physical_dims_preserved"]) if "physical_dims_preserved" in df else True
    all_origins_preserved = all_true(df["origin_preserved"]) if "origin_preserved" in df else True
    all_directions_preserved = all_true(df["direction_preserved"]) if "direction_preserved" in df else True
    mask_df = df[df["is_label"] == True]
    all_labels_preserved_for_masks = all_true(mask_df["labels_preserved"]) if not mask_df.empty else True

    print("\nOverall validation flags:")
    print(f" - all_isotropic: {all_isotropic}")
    print(f" - all_dims_preserved: {all_dims_preserved}")
    print(f" - all_origins_preserved: {all_origins_preserved}")
    print(f" - all_directions_preserved: {all_directions_preserved}")
    print(f" - all_labels_preserved_for_masks: {all_labels_preserved_for_masks}")

    # Save CSV
    csv_path = output_root / "resampling_1mm_validation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved validation results CSV to: {csv_path}")

    print(f"\nProcessed intensity files: {n_images}")
    print(f"Processed mask files: {n_masks}")


if __name__ == "__main__":
    main()
