"""
Intra-Subject Co-registration Script

This script performs intra-subject co-registration on MRI images using ANTs.
For each patient in the input directory, it selects a fixed image (preferably a T1)
and registers all other images to it.

Usage:
    python 4-intra_subject_registration.py \
        --input_dir repeat-1 \
        --output_dir repeat-4

If no arguments are provided, the script will default to the historical paths
used in prior runs (25-nov-new-images-output-denoised-bfc → 25-nov-registered-new).

The registered images are saved in a new directory with "-reg" added to their filenames.
"""

import os
import argparse
import ants
from helpers import add_suffix_to_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CLI to allow flexible input/output directories while keeping historical defaults
parser = argparse.ArgumentParser(description='Intra-subject co-registration of MRI volumes using ANTs.')
parser.add_argument('--input_dir', type=str, default=os.path.join(BASE_DIR, '25-nov-new-images-output-denoised-bfc'),
                    help='Root input directory containing patient subfolders. Default: 25-nov-new-images-output-denoised-bfc')
parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_DIR, '25-nov-registered-new'),
                    help='Root output directory to save registered images. Default: 25-nov-registered-new')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Validate input directory and get list of patient folders
if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

patient_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

# Process each patient
for patient in patient_folders:
    print(f"Processing {patient}...")

    # Create patient output directory
    patient_output_dir = os.path.join(output_dir, patient)
    if not os.path.exists(patient_output_dir):
        os.makedirs(patient_output_dir)

    # Determine fixed image for this patient.
    # Prefer an image containing 't1' (case-insensitive). Otherwise, pick the first .nii/.nii.gz found.
    patient_input_dir = os.path.join(input_dir, patient)
    all_files = [f for f in os.listdir(patient_input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    if not all_files:
        print(f"  Warning: No NIfTI images found for {patient}, skipping...")
        continue

    # Try to select a T1-like image as fixed
    # Prefer specifically the denoised+bias-corrected T1 if available
    preferred_order = [
        't1-d-bf.nii.gz', 't1-d-bf.nii',
        't1c-d-bf.nii.gz', 't1c-d-bf.nii',
        't1.nii.gz', 't1.nii'
    ]
    fixed_image_name = None
    for cand in preferred_order:
        if cand in all_files:
            fixed_image_name = cand
            break
    # If none of the strict candidates are present, fall back to any filename containing 't1'
    t1_candidates = [f for f in all_files if 't1' in f.lower()]
    if fixed_image_name is None:
        fixed_image_name = t1_candidates[0] if t1_candidates else all_files[0]
    fixed_image_path = os.path.join(patient_input_dir, fixed_image_name)

    # Load fixed image
    fixed_image = ants.image_read(fixed_image_path, reorient='IAL')

    # Copy fixed image to output directory
    fixed_image_output_path = os.path.join(patient_output_dir, fixed_image_name)
    ants.image_write(fixed_image, fixed_image_output_path)
    print(f"  Copied fixed image to output directory")

    # Get all other images in the patient folder
    image_files = [f for f in all_files if (f.endswith('.nii') or f.endswith('.nii.gz')) and f != fixed_image_name]

    # Register each image to the fixed image
    for image_file in image_files:
        moving_image_path = os.path.join(patient_input_dir, image_file)

        # Load moving image
        moving_image = ants.image_read(moving_image_path, reorient='IAL')

        print(f"  Registering {image_file} to {fixed_image_name}...")

        # Perform registration
        registration_results = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='Affine',
            initial_transform=None,
            aff_metric='mattes',
            verbose=False  # Set to True for detailed registration output
        )

        # Get the registered image
        registered_image = registration_results['warpedmovout']

        # Create output filename with -reg suffix
        output_filename = add_suffix_to_filename(image_file, "-reg")
        output_path = os.path.join(patient_output_dir, output_filename)

        # Save registered image
        ants.image_write(registered_image, output_path)
        print(f"  Saved registered image as {output_filename}")

print("Registration complete for all patients!")
