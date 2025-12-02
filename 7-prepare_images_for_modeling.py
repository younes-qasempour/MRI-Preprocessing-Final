#!/usr/bin/env python3
"""
Create a new directory named Images-For-Modeling that mirrors the structure of
30-nov-Final-MRI-Data-1mm/Images, with one subdirectory per patient and a
standardized set of MRI filenames per patient.

Each copied image is renamed to a short, consistent format that includes the
patient ID and the MRI sequence, for example:
  Patient01_T1.nii.gz
  Patient01_T1C.nii.gz
  Patient01_T2.nii.gz
  Patient01_FLAIR.nii.gz

Supported input naming patterns (observed in the project):
  - resampled_1mm_t1_se_tra-FS-d-bf_brain-extracted.nii.gz            -> T1
  - resampled_1mm_t1_se_tra-FSGD-d-bf-reg_brain-extracted.nii.gz      -> T1C
  - resampled_1mm_t2_tirm_tra_dark-fluid_320-d-bf-reg_brain-extracted -> FLAIR
  - resampled_1mm_t2_tse_tra-d-bf-reg_brain-extracted.nii.gz          -> T2
  - resampled_1mm_t1-d-bf_brain-extracted.nii.gz                       -> T1
  - resampled_1mm_t1c-d-bf-reg_brain-extracted.nii.gz                  -> T1C
  - resampled_1mm_t2-d-bf-reg_brain-extracted.nii.gz                   -> T2
  - resampled_1mm_f-d-bf-reg_brain-extracted.nii.gz                    -> FLAIR

If multiple candidates exist for the same sequence in a patient, the script
prefers registered images (filenames containing "-reg_") over non-registered.

This script makes no destructive changes: it only copies files.
"""

from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# Defaults updated to consume repeat-6 outputs and write to final-repeat
DEFAULT_INPUT_ROOT = os.path.join(REPO_ROOT, 'repeat-6', 'Images')
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, 'final-repeat', 'Images-For-Modeling')


def is_nifti(filename: str) -> bool:
    return filename.endswith('.nii.gz') or filename.endswith('.nii')


def detect_sequence(filename: str) -> Optional[str]:
    """Detect standardized sequence label from filename.

    Returns one of {"T1", "T1C", "T2", "FLAIR"} or None if not recognized.
    """
    f = filename.lower()

    # FLAIR first: long pattern uses tirm, short pattern uses 'f'
    if ('tirm' in f) or re.search(r'[\/_\-]f[\-_]', f):
        return 'FLAIR'

    # T1C (post-contrast): examples include 't1c' or 'fsgd'
    if ('t1c' in f) or ('fsgd' in f) or ('t1_gd' in f) or ('t1gd' in f):
        return 'T1C'

    # T2 (ensure not flair which also includes 't2')
    if ('t2' in f) and ('tirm' not in f):
        return 'T2'

    # T1 (ensure not t1c)
    if ('t1' in f) and ('t1c' not in f) and ('fsgd' not in f):
        return 'T1'

    return None


def candidate_priority(filename: str) -> int:
    """Rank candidates for the same sequence. Higher is better.

    Preference: registered > non-registered; keep original otherwise.
    """
    f = filename.lower()
    score = 0
    if '-reg_' in f or '-reg.' in f or f.endswith('-reg'):
        score += 10
    # Explicit brain-extracted (they all are, but keep minor weight)
    if 'brain-extracted' in f:
        score += 1
    return score


def collect_patient_sequences(patient_dir: str) -> Dict[str, List[str]]:
    seq_map: Dict[str, List[str]] = defaultdict(list)
    for name in os.listdir(patient_dir):
        if not is_nifti(name):
            continue
        seq = detect_sequence(name)
        if seq is None:
            continue
        seq_map[seq].append(os.path.join(patient_dir, name))
    return seq_map


def choose_best(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    return sorted(candidates, key=candidate_priority, reverse=True)[0]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def standard_name(patient_id: str, sequence: str, ext: str) -> str:
    # Always use .nii.gz if source is .nii.gz else .nii
    normalized_seq = sequence.upper()
    return f"{patient_id}_{normalized_seq}{ext}"


def run(input_root: str = DEFAULT_INPUT_ROOT, output_root: str = DEFAULT_OUTPUT_ROOT) -> None:
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    ensure_dir(output_root)

    patients = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    patients.sort()

    for patient in patients:
        src_dir = os.path.join(input_root, patient)
        dst_dir = os.path.join(output_root, patient)
        ensure_dir(dst_dir)

        seq_map = collect_patient_sequences(src_dir)

        for sequence in ['T1', 'T1C', 'T2', 'FLAIR']:
            best = choose_best(seq_map.get(sequence, []))
            if best is None:
                # Sequence missing for this patient; skip silently but keep structure
                continue

            ext = '.nii.gz' if best.endswith('.nii.gz') else '.nii'
            dst_name = standard_name(patient, sequence, ext)
            dst_path = os.path.join(dst_dir, dst_name)

            # Copy with metadata preserved where possible
            shutil.copy2(best, dst_path)

    print(f"Done. Output written to: {output_root}")


if __name__ == '__main__':
    # Allow overriding paths via CLI while keeping backward-compatible defaults
    import argparse
    parser = argparse.ArgumentParser(description='Prepare images for modeling (copy and standardize names).')
    parser.add_argument('--input-root', type=str, default=DEFAULT_INPUT_ROOT,
                        help='Input root directory containing per-patient image folders (default: repeat-6/Images).')
    parser.add_argument('--output-root', type=str, default=DEFAULT_OUTPUT_ROOT,
                        help='Output root directory for standardized images (default: final-repeat/Images-For-Modeling).')
    args = parser.parse_args()
    run(input_root=args.input_root, output_root=args.output_root)
