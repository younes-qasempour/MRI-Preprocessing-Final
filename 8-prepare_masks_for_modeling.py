#!/usr/bin/env python3
"""
Create a new directory named Masks-For-Modeling that contains ALL masks in a
single, flat folder (no per-patient subfolders).

Input observed pattern (source is a flat directory, one file per patient):
  resampled_1mm_GBM001.nii.gz.nii.seg.nrrd
  resampled_1mm_GBM003.nii.gz.nii.seg.nrrd
  ...

Output structure and naming (flat):
  Masks-For-Modeling/
    Patient01_MASK.seg.nrrd
    Patient03_MASK.seg.nrrd
    ...

Notes:
  - We keep the original .seg.nrrd format (no conversion), only copying.
  - Patient ID is derived from the GBM number (e.g., GBM001 -> Patient01).
  - If this script was previously used and created per-patient subfolders, it
    now automatically flattens that structure by moving files up to the root
    and removing empty subfolders.
  - Non-destructive with respect to the INPUT directory; output files may be
    overwritten to keep idempotent results.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional, Tuple


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(REPO_ROOT, '30-nov-Final-MRI-Data-1mm', 'Masks')
OUTPUT_ROOT = os.path.join(REPO_ROOT, 'Masks-For-Modeling')


GBM_RE = re.compile(r"gbm\s*(\d{1,3})", re.IGNORECASE)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def flatten_existing_output(output_root: str) -> int:
    """Move any files from subdirectories up into the output root and
    remove emptied subdirectories. Returns the number of files moved.

    This helps migrate from an older per-patient layout to the new flat layout.
    """
    if not os.path.isdir(output_root):
        return 0

    moved = 0
    for entry in list(os.scandir(output_root)):
        if not entry.is_dir():
            continue
        subdir = entry.path
        for dirpath, _, filenames in os.walk(subdir):
            for fn in filenames:
                src = os.path.join(dirpath, fn)
                dst = os.path.join(output_root, fn)
                # If a file with the same name exists at root, overwrite to keep
                # deterministic latest result.
                shutil.move(src, dst)
                moved += 1
        # try to remove emptied dir tree
        try:
            # remove bottom-up
            for root, dirs, files in list(os.walk(subdir, topdown=False)):
                for f in files:
                    # Should be none left; safeguard in case of race
                    fp = os.path.join(root, f)
                    if os.path.exists(fp):
                        os.remove(fp)
                for d in dirs:
                    dp = os.path.join(root, d)
                    if os.path.isdir(dp):
                        os.rmdir(dp)
            if os.path.isdir(subdir):
                os.rmdir(subdir)
        except OSError:
            # If deletion fails, skip silently
            pass
    return moved


def extract_patient_id(name: str) -> Optional[str]:
    """Return normalized PatientID (e.g., 'Patient01') from a mask filename.

    Accepts various placements/cases of 'GBM###'.
    """
    m = GBM_RE.search(name)
    if not m:
        return None
    num = int(m.group(1))
    # Normalize to two-digit for 1..99, else use natural (e.g., 100 -> '100')
    if num < 100:
        pid = f"Patient{num:02d}"
    else:
        pid = f"Patient{num}"
    return pid


def is_mask_file(filename: str) -> bool:
    # Keep it simple: accept .seg.nrrd files only (as observed)
    f = filename.lower()
    return f.endswith('.seg.nrrd') or f.endswith('.nrrd')


def target_name(patient_id: str, src_name: str) -> str:
    # Preserve the .seg.nrrd extension if present, otherwise keep the extension of the source
    src_lower = src_name.lower()
    if src_lower.endswith('.seg.nrrd'):
        ext = '.seg.nrrd'
    elif src_lower.endswith('.nrrd'):
        ext = '.nrrd'
    else:
        # Fallback: keep everything after the first dot
        ext = os.path.splitext(src_name)[1]
    return f"{patient_id}_MASK{ext}"


def run(input_root: str = INPUT_ROOT, output_root: str = OUTPUT_ROOT) -> None:
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    ensure_dir(output_root)

    # Flatten any legacy per-patient structure from previous runs.
    moved_legacy = flatten_existing_output(output_root)

    entries = sorted(os.listdir(input_root))

    num_copied = 0
    num_skipped = 0

    for name in entries:
        if not is_mask_file(name):
            num_skipped += 1
            continue

        patient_id = extract_patient_id(name)
        if not patient_id:
            num_skipped += 1
            continue

        src_path = os.path.join(input_root, name)
        dst_name = target_name(patient_id, name)
        dst_path = os.path.join(output_root, dst_name)

        # Overwrite if exists to make the process idempotent and ensure flat layout
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy2(src_path, dst_path)
        num_copied += 1

    summary_extra = f" Flattened {moved_legacy} legacy file(s)." if moved_legacy else ""
    print(f"Done. Copied {num_copied} mask(s); skipped {num_skipped}.{summary_extra} Output: {output_root}")


if __name__ == '__main__':
    run()
