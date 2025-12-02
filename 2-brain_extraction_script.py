import os
import glob
import argparse
import ants
from antspynet.utilities import brain_extraction
from helpers import add_suffix_to_filename

# Force CPU usage to avoid GPU-related errors
# The issue description mentioned to ignore warnings and run as is
# This will help us avoid cuDNN library version mismatch errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CLI to allow flexible input/output directories while keeping historical defaults
parser = argparse.ArgumentParser(description='Run brain extraction to generate brain masks for MRI volumes.')
parser.add_argument('--input_dir', type=str, default=os.path.join(BASE_DIR, '25-nov-registered-new'),
                    help='Root input directory containing patient subfolders. Default: 25-nov-registered-new')
parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_DIR, '26-nov-brain-masks-new'),
                    help='Root output directory to save brain masks. Default: 26-nov-brain-masks-new')
args = parser.parse_args()

input_dir = args.input_dir
output_masks_dir = args.output_dir

# Validate input
if not os.path.isdir(input_dir):
    raise SystemExit(f"Input directory not found: {input_dir}")

# Create output directory if it doesn't exist
os.makedirs(output_masks_dir, exist_ok=True)

# Define the modality string for brain extraction
modality_string = "t1"

# Get all patient folders
patient_folders = [p for p in glob.glob(os.path.join(input_dir, 'Patient*')) if os.path.isdir(p)]
print(f'Found {len(patient_folders)} patient folders in {input_dir}')

# Process each patient folder
for patient_folder in patient_folders:
    patient_name = os.path.basename(patient_folder)
    print(f'Processing {patient_name}...')

    # Define the input file path (try common T1 options in priority order)
    candidate_filenames = [
        't1-d-bf-reg.nii.gz',   # preferred: registered T1 if available
        't1-d-bf.nii.gz',       # fallback: non-registered T1
        't1c-d-bf-reg.nii.gz',  # contrast-enhanced T1 (less preferred)
        'f-d-bf-reg.nii.gz',    # FLAIR (as last resort)
        't2-d-bf-reg.nii.gz'    # T2 (as last resort)
    ]

    input_file = None
    for cand in candidate_filenames:
        cand_path = os.path.join(patient_folder, cand)
        if os.path.exists(cand_path):
            input_file = cand_path
            break

    # Check if the input file exists
    if input_file is None or not os.path.exists(input_file):
        print(f'Warning: {input_file} does not exist. Skipping...')
        continue

    # Create output folder for this patient
    # Create output folder for this patient (masks only)
    patient_masks_dir = os.path.join(output_masks_dir, patient_name)
    os.makedirs(patient_masks_dir, exist_ok=True)

    # Read the input image
    print(f'Reading {input_file}...')
    input_image = ants.image_read(input_file, reorient='IAL')

    # Extract brain mask
    print(f'Extracting brain mask...')
    try:
        # Primary method: deep learning brain extraction (requires internet on first run to fetch weights)
        prob_brain_mask = brain_extraction(input_image, modality=modality_string, verbose=True)
        # Get binary mask from probability map
        brain_mask = ants.get_mask(prob_brain_mask, low_thresh=0.5)
    except Exception as e:
        # Fallback: offline classical mask estimation directly from the input image
        print(f'Warning: brain_extraction failed with error: {e}')
        print('Falling back to classical intensity-based mask using ants.get_mask().')
        # cleanup performs morphological opening/closing; component_threshold keeps largest component
        brain_mask = ants.get_mask(input_image, cleanup=2)

    # Define output file path
    # Build output filenames based on input
    base_filename = os.path.basename(input_file)
    mask_filename = add_suffix_to_filename(base_filename, 'brainMask')
    output_mask_file = os.path.join(patient_masks_dir, mask_filename)

    # Save the brain mask
    print(f'Saving brain mask to {output_mask_file}...')
    brain_mask.to_file(output_mask_file)

    print(f'Completed processing {patient_name}')

print('All patients processed successfully!')
