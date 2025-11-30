import numpy as np
import nibabel as nib
import nrrd
import os
import glob
from pathlib import Path


def load_nifti_image(file_path):
    """
    Load a NIfTI image file and return the image object and data.

    Args:
        file_path (str): Path to the NIfTI image file

    Returns:
        tuple: (nifti_img, image_data) - The NIfTI image object and its data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI image file not found: {file_path}")

    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()
    return nifti_img, image_data


def load_brain_mask(file_path):
    """
    Load a brain mask file (NRRD or NIfTI) and convert it to boolean.

    Args:
        file_path (str): Path to the brain mask file

    Returns:
        numpy.ndarray: Boolean brain mask data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Brain mask file not found: {file_path}")

    lower = file_path.lower()
    if lower.endswith('.nrrd'):
        brain_mask_data, _ = nrrd.read(file_path)
    elif lower.endswith('.nii') or lower.endswith('.nii.gz'):
        mask_img = nib.load(file_path)
        brain_mask_data = mask_img.get_fdata()
    else:
        # Try NRRD as default fallback
        try:
            brain_mask_data, _ = nrrd.read(file_path)
        except Exception:
            # Then try NIfTI
            mask_img = nib.load(file_path)
            brain_mask_data = mask_img.get_fdata()

    # Convert to boolean (handle label masks with values >0)
    return (brain_mask_data > 0).astype(bool)


def z_score_normalize_brain(image_data, brain_mask_data):
    """
    Perform Z-score normalization on brain voxels and set non-brain voxels to 0.

    Args:
        image_data (numpy.ndarray): The original image data
        brain_mask_data (numpy.ndarray): Boolean brain mask

    Returns:
        numpy.ndarray: Z-score normalized image data with non-brain voxels set to 0
    """
    # Select voxels within the brain mask
    brain_voxels = image_data[brain_mask_data]

    if brain_voxels.size == 0:
        print("Warning: Brain mask is empty. Cannot perform Z-score normalization.")
        return image_data  # Return original data if mask is empty

    # Calculate mean and standard deviation of brain voxels
    mean_val = np.mean(brain_voxels)
    std_val = np.std(brain_voxels)

    # Create a copy of the image data for normalization
    normalized_data = np.copy(image_data)

    if std_val == 0:  # Avoid division by zero
        print("Warning: Standard deviation of brain voxels is zero. Performing mean centering only.")
        normalized_data[brain_mask_data] = normalized_data[brain_mask_data] - mean_val
    else:
        # Apply Z-score normalization to brain voxels only
        normalized_data[brain_mask_data] = (normalized_data[brain_mask_data] - mean_val) / std_val

    # Set non-brain voxels (outside the mask) to 0
    normalized_data[~brain_mask_data] = 0

    return normalized_data


def save_normalized_image(normalized_data, original_img, output_path):
    """
    Save the normalized data as a new NIfTI image.

    Args:
        normalized_data (numpy.ndarray): The normalized image data
        original_img (nibabel.nifti1.Nifti1Image): The original NIfTI image object
        output_path (str): Path where the normalized image will be saved
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new NIfTI image with the normalized data
    normalized_nifti_img = nib.Nifti1Image(normalized_data, original_img.affine, original_img.header)

    # Save the normalized image
    nib.save(normalized_nifti_img, output_path)
    print(f"Normalized image saved to: {output_path}")


def process_patient(patient_dir, mask_dir, output_base_dir):
    """
    Process all images for a single patient.

    Args:
        patient_dir (str): Path to the patient's directory with brain-extracted images
        mask_dir (str): Path to the patient's directory with brain mask
        output_base_dir (str): Base directory for output
    
    Returns:
        tuple: (success_count, total_count) - Number of successfully processed images and total images
    """
    patient_id = os.path.basename(patient_dir)
    print(f"\nProcessing {patient_id}...")
    
    # Create output directory for this patient
    output_patient_dir = os.path.join(output_base_dir, patient_id)
    if not os.path.exists(output_patient_dir):
        os.makedirs(output_patient_dir)
    
    # Find all NIfTI files for this patient
    nifti_files = glob.glob(os.path.join(patient_dir, "*.nii.gz"))
    
    # Find the mask file for this patient (support .nrrd and .nii/.nii.gz)
    mask_files = []
    mask_files.extend(glob.glob(os.path.join(mask_dir, "*.nrrd")))
    mask_files.extend(glob.glob(os.path.join(mask_dir, "*.nii")))
    mask_files.extend(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
    
    if not mask_files:
        print(f"Error: No mask file found for {patient_id}")
        return 0, len(nifti_files)
    
    # Use the first mask file (as per the issue description, each patient has one mask for all images)
    mask_path = mask_files[0]
    
    success_count = 0
    
    # Process each NIfTI file
    for nifti_path in nifti_files:
        try:
            # Get the base filename
            base_filename = os.path.basename(nifti_path)
            
            # Define output path
            output_path = os.path.join(output_patient_dir, base_filename)
            
            print(f"Processing: {base_filename}")
            
            # Load the NIfTI image
            nifti_img, image_data = load_nifti_image(nifti_path)
            
            # Load the brain mask
            brain_mask_data = load_brain_mask(mask_path)
            
            # Check if mask and image dimensions match
            if brain_mask_data.shape != image_data.shape:
                print(f"Warning: Mask shape {brain_mask_data.shape} does not match image shape {image_data.shape} for {base_filename}")
                continue
            
            # Perform Z-score normalization
            normalized_data = z_score_normalize_brain(image_data, brain_mask_data)
            
            # Save the normalized image
            save_normalized_image(normalized_data, nifti_img, output_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {nifti_path}: {e}")
    
    return success_count, len(nifti_files)


def main():
    """
    Main function to process all patients and their images.
    """
    # Define base directories (updated to 27/26 Nov folders)
    input_base_dir = "27-nov-brains-extracted-new"
    mask_base_dir = "26-nov-brain-masks-new-modified"
    output_base_dir = "27-nov-intensity-normalized-new"
    
    # Create output base directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Get all patient directories (any subdirectory under input_base_dir)
    patient_dirs = [
        os.path.join(input_base_dir, d)
        for d in os.listdir(input_base_dir)
        if os.path.isdir(os.path.join(input_base_dir, d))
    ]
    
    total_patients = len(patient_dirs)
    total_images = 0
    total_success = 0
    
    print(f"Found {total_patients} patient directories to process.")
    
    # Process each patient
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        mask_dir = os.path.join(mask_base_dir, patient_id)
        
        if not os.path.exists(mask_dir):
            print(f"Error: Mask directory not found for {patient_id}")
            continue
        
        success_count, image_count = process_patient(patient_dir, mask_dir, output_base_dir)
        total_images += image_count
        total_success += success_count
    
    # Print summary
    print("\n" + "="*50)
    print("INTENSITY NORMALIZATION SUMMARY")
    print("="*50)
    print(f"Total patients processed: {total_patients}")
    print(f"Total images processed: {total_success}/{total_images}")
    if total_images > 0:
        print(f"Success rate: {(total_success/total_images)*100:.2f}%")
    else:
        print("Success rate: N/A (no images found)")
    print("="*50)


if __name__ == "__main__":
    main()