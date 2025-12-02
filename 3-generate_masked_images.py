#!/usr/bin/env python3
"""
Script to generate masked MRI images using existing brain masks.

This script (current workflow as per issue request):
1. Reads MRI images from repeat-4
2. Reads brain masks (NRRD format) from repeat-2
3. Applies masks to images
4. Saves masked images to repeat-3 with the same structure

The script includes error handling for various edge cases and will skip
already processed images if run multiple times.
"""

import os
import glob
import ants
import SimpleITK as sitk
import numpy as np
import sys
from helpers import add_suffix_to_filename

# Define paths (updated for repeat workflow)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "repeat-4")
MASKS_DIR = os.path.join(BASE_DIR, "repeat-2")
OUTPUT_DIR = os.path.join(BASE_DIR, "repeat-3")

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_patient_directories():
    """Get list of patient directories."""
    return [d for d in os.listdir(IMAGES_DIR) 
            if os.path.isdir(os.path.join(IMAGES_DIR, d))]

def process_patient(patient_dir):
    """Process all images for a single patient."""
    print(f"Processing {patient_dir}...")
    
    try:
        # Create output directory for this patient
        patient_output_dir = os.path.join(OUTPUT_DIR, patient_dir)
        ensure_directory_exists(patient_output_dir)
        
        # Get all image files for this patient
        image_files = glob.glob(os.path.join(IMAGES_DIR, patient_dir, "*.nii.gz"))
        
        if not image_files:
            print(f"No image files found for {patient_dir}")
            return
        
        # Get the mask file for this patient (NRRD format)
        # Be flexible on naming; pick the first .nrrd mask found in the patient's folder
        mask_files = glob.glob(os.path.join(MASKS_DIR, patient_dir, "*.nrrd"))
        
        if not mask_files:
            print(f"No mask file found for {patient_dir}")
            return
        
        # We expect only one mask file per patient
        mask_file = mask_files[0]
        
        # Read the mask (NRRD). Prefer ANTs reader; if it fails, use SimpleITK and convert.
        print(f"Reading mask: {mask_file}")
        try:
            try:
                mask_ants = ants.image_read(mask_file, reorient='IAL')
            except Exception:
                # Fallback to SimpleITK then convert to ANTs image
                mask_sitk = sitk.ReadImage(mask_file)
                mask_np = sitk.GetArrayFromImage(mask_sitk)  # z, y, x (ITK)
                # Convert ITK (z,y,x) to ANTs (x,y,z) expected orientation via transpose
                mask_np_ants = np.transpose(mask_np, (2, 1, 0))
                spacing = tuple(reversed(mask_sitk.GetSpacing()))
                direction = mask_sitk.GetDirection()
                # ITK direction is a flat tuple length 9; reshape then reverse axes to match transpose
                if len(direction) == 9:
                    direction_matrix = np.array(direction).reshape(3, 3)
                    direction_matrix = direction_matrix[:, ::-1][::-1, :]  # approximate axis flip
                    direction_tuple = tuple(direction_matrix.flatten())
                else:
                    direction_tuple = direction
                origin = tuple(reversed(mask_sitk.GetOrigin()))
                mask_ants = ants.from_numpy(mask_np_ants, origin=origin, spacing=spacing, direction=direction_tuple)
            # Binarize mask to ensure proper masking even if labels > 1
            mask_ants = ants.threshold_image(mask_ants, 0.5, 1e9, 1, 0)
        except Exception as e:
            print(f"Error reading mask file {mask_file}: {str(e)}")
            return
        
        # Process each image file
        for image_file in image_files:
            image_filename = os.path.basename(image_file)
            
            # Create output filename
            output_filename = add_suffix_to_filename(image_filename, "brain-extracted")
            output_path = os.path.join(patient_output_dir, output_filename)
            
            # Skip if output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {image_filename} - output file already exists")
                continue
                
            print(f"Processing image: {image_filename}")
            
            try:
                # Read the image using ANTs
                image_ants = ants.image_read(image_file, reorient='IAL')

                # If sizes don't match due to minor header/orientation differences, resample mask to image space
                if image_ants.shape != mask_ants.shape:
                    try:
                        mask_for_image = ants.resample_image_to_target(mask_ants, image_ants, interp='nearestNeighbor')
                    except Exception as re:
                        print(f"  Warning: resampling mask to image space failed ({re}). Skipping {image_filename}.")
                        continue
                else:
                    mask_for_image = mask_ants

                # Apply the mask to the image
                masked_image = ants.mask_image(image_ants, mask_for_image)

                # Save the masked image
                print(f"Saving masked image to: {output_path}")
                masked_image.to_file(output_path)
            except Exception as e:
                print(f"Error processing image {image_file}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error processing patient {patient_dir}: {str(e)}")
        return

def main():
    """Main function to process all patients."""
    print("Starting brain extraction process...")
    
    try:
        # Ensure output directory exists
        ensure_directory_exists(OUTPUT_DIR)
        
        # Get all patient directories
        try:
            patient_dirs = get_patient_directories()
            print(f"Found {len(patient_dirs)} patient directories")
            
            if not patient_dirs:
                print("No patient directories found. Please check the input directory.")
                return
                
        except Exception as e:
            print(f"Error getting patient directories: {str(e)}")
            return
        
        # Process each patient
        processed_count = 0
        error_count = 0
        
        for patient_dir in patient_dirs:
            try:
                process_patient(patient_dir)
                processed_count += 1
            except Exception as e:
                print(f"Error processing patient {patient_dir}: {str(e)}")
                error_count += 1
        
        # Print summary
        print("\nBrain extraction process summary:")
        print(f"Total patients: {len(patient_dirs)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Errors: {error_count}")
        
        if error_count == 0:
            print("\nBrain extraction completed successfully!")
        else:
            print("\nBrain extraction completed with some errors. Check the log for details.")
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Brain extraction process failed.")

if __name__ == "__main__":
    main()