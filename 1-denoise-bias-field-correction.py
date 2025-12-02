#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to automatically perform denoising and bias field correction on MRI images.

Usage:
    python 1-denoise-bias-field-correction.py \
        --input_dir MRI-Repeat \
        --output_dir repeat-1

If no arguments are provided, the script will default to the historical paths
used in prior runs (25-nov-New-MRI-GBM-Thesis → 25-nov-new-images-output-denoised-bfc).
"""

import os
import sys
import argparse
import SimpleITK as sitk
from helpers import add_suffix_to_filename
import warnings

# Force CPU usage to avoid potential GPU/cudnn mismatches as used in previous scripts
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

def denoise_image(image, time_step=0.03125, num_iterations=5):
    """
    Denoise an image using curvature flow.
    
    Args:
        image: SimpleITK image to denoise
        time_step: Time step for the denoising algorithm (smaller for more stability)
        num_iterations: Number of iterations for the denoising algorithm
        
    Returns:
        Denoised SimpleITK image
    """
    denoiser = sitk.CurvatureFlowImageFilter()
    denoiser.SetTimeStep(time_step)
    denoiser.SetNumberOfIterations(num_iterations)
    return denoiser.Execute(image)

def bias_field_correction(denoised_image):
    """
    Perform bias field correction on a denoised image.
    
    Args:
        denoised_image: Denoised SimpleITK image
        
    Returns:
        Bias field corrected SimpleITK image
    """
    # Create a mask for the head
    transformed = sitk.RescaleIntensity(denoised_image, 0, 255)
    transformed = sitk.LiThreshold(transformed, 0, 1)
    head_mask = transformed
    
    # Shrink the image for faster processing
    shrinkFactor = 2
    inputImage = sitk.Shrink(denoised_image, [shrinkFactor] * denoised_image.GetDimension())
    maskImage = sitk.Shrink(head_mask, [shrinkFactor] * denoised_image.GetDimension())
    
    # Apply bias field correction
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)
    
    # Get the bias field and apply it to the original resolution image
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(denoised_image)
    exp_bias_field = sitk.Cast(sitk.Exp(log_bias_field), denoised_image.GetPixelID())
    
    # Perform the division to get the corrected image
    corrected_img_full_res = denoised_image / exp_bias_field
    
    return corrected_img_full_res

def process_image(input_path, output_path):
    """
    Process a single MRI image with denoising and bias field correction.
    
    Args:
        input_path: Path to the input image
        output_path: Path to save the processed image
    """
    try:
        # Read the image
        raw_img = sitk.ReadImage(input_path, sitk.sitkFloat32)
        
        # Denoise the image
        denoised_img = denoise_image(raw_img)
        
        # Apply bias field correction
        corrected_img = bias_field_correction(denoised_img)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed (denoised + bias-corrected) image
        sitk.WriteImage(corrected_img, output_path)
        
        print(f"Processed (denoised + bias-corrected): {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def main():
    """
    Main function to process all MRI images in a dataset directory structure.
    Expects input_dir to contain per-patient folders with NIfTI files inside.
    """
    # Get the base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Denoise and perform bias field correction on MRI volumes.')
    parser.add_argument('--input_dir', type=str, default=os.path.join(BASE_DIR, '25-nov-New-MRI-GBM-Thesis'),
                        help='Path to the root input directory containing patient subfolders (default: 25-nov-New-MRI-GBM-Thesis)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_DIR, '25-nov-new-images-output-denoised-bfc'),
                        help='Path to the root output directory where processed images will be saved (default: 25-nov-new-images-output-denoised-bfc)')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Validate input directory exists
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the input folder
    for patient_folder in os.listdir(input_dir):
        patient_path = os.path.join(input_dir, patient_folder)

        # Skip if not a directory
        if not os.path.isdir(patient_path):
            continue

        # Create patient output directory
        patient_output_path = os.path.join(output_dir, patient_folder)
        os.makedirs(patient_output_path, exist_ok=True)

        # Process all images in the patient folder
        for image_file in os.listdir(patient_path):
            # Skip if not a NIfTI file
            if not (image_file.endswith('.nii') or image_file.endswith('.nii.gz')):
                continue

            # Define input and output paths
            input_path = os.path.join(patient_path, image_file)

            # Add suffix to the filename
            output_filename = add_suffix_to_filename(image_file, suffix='-d-bf')
            output_path = os.path.join(patient_output_path, output_filename)

            # Process the image
            process_image(input_path, output_path)

if __name__ == "__main__":
    main()
    print("Processing complete!")