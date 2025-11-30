#!/usr/bin/env python3
"""
Script to generate masked MRI images using existing brain masks.

This script:
1. Reads MRI images from 25-nov-registered-new
2. Reads brain masks from 26-nov-brain-masks-new-modified
3. Applies masks to images
4. Saves masked images to 27-nov-brains-extracted-new with the same structure

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

# Define paths (update for 25/26/27-nov workflow)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "25-nov-registered-new")
MASKS_DIR = os.path.join(BASE_DIR, "26-nov-brain-masks-new-modified")
OUTPUT_DIR = os.path.join(BASE_DIR, "27-nov-brains-extracted-new")

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
        
        # Get the mask file for this patient - using more specific pattern
        mask_files = glob.glob(os.path.join(MASKS_DIR, patient_dir, "*_brainMask*.nrrd"))
        
        if not mask_files:
            print(f"No mask file found for {patient_dir}")
            return
        
        # We expect only one mask file per patient
        mask_file = mask_files[0]
        
        # Read the mask using ANTs
        print(f"Reading mask: {mask_file}")
        try:
            mask_ants = ants.image_read(mask_file, reorient='IAL')
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