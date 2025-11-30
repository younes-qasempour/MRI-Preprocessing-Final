MRI-Preprocessing-Final

Overview
- This repository contains a practical MRI preprocessing pipeline used for glioblastoma and brain MRI volumes. It provides step-by-step scripts for denoising, bias-field correction, brain extraction, masking, intra-subject registration, intensity normalization, and resampling to 1 mm isotropic resolution.

Key Features
- Scripted, reproducible preprocessing steps
- Works with NIfTI (.nii/.nii.gz) and NRRD mask files
- Uses SimpleITK, ANTs/ANTsPy, and antspynet
- Utilities for visualization and filename handling

Repository Structure
- 1-denoise-bias-field-correction.py: Denoising + N4 bias field correction
- 2-brain_extraction_script.py: Brain mask generation (antspynet with classical fallback)
- 3-generate_masked_images.py: Apply brain masks to images
- 4-intra_subject_registration.py: Register modalities of a patient (open this file for exact details)
- 5-intensity_normalization_batch.py: Batch intensity normalization
- 6-resample_to_1mm_isotropic.py: Resample images to 1 mm isotropic spacing
- helpers.py: Helper utilities (visualization, filename suffixing, info)
- environment.yml: Conda environment definition
- .gitignore: Git exclusions

Requirements
- OS: Linux, macOS, or Windows (WSL recommended on Windows)
- Python: 3.9+ (see environment.yml)
- Recommended: Conda (miniforge/miniconda/mamba)

Quick Setup (Conda)
1) Create and activate the environment
   conda env create -f environment.yml
   conda activate mri-preprocessing-final

2) If any packages are missing, install via pip
   pip install antspyx antspynet SimpleITK opencv-python matplotlib ipywidgets numpy

Data Organization
The scripts expect patient-organized folders under project-root. Examples (some folders may not exist in your clone):
- 25-nov-New-MRI-GBM-Thesis/PatientXX/*.nii.gz
- 25-nov-registered-new/PatientXX/*.nii.gz
- 26-nov-brain-masks-new(-modified)/PatientXX/*_brainMask*.nrrd
- 27-nov-brains-extracted-new/PatientXX/*.nii.gz
- 30-nov-Final-MRI-Data and 30-nov-Final-MRI-Data-1mm are example end-results

Usage: Step-by-Step
Important: Most scripts are designed to be run from the repository root.

1) Denoise + Bias Field Correction
   python 1-denoise-bias-field-correction.py
   - Reads from: 25-nov-New-MRI-GBM-Thesis/Patient*/
   - Writes to: 25-nov-new-images-output-denoised-bfc/Patient*/ with “-d-bf” suffix

2) Brain Extraction (Mask)
   python 2-brain_extraction_script.py
   - Reads from: 25-nov-registered-new/Patient*/ (tries t1*-reg first)
   - Writes masks to: 26-nov-brain-masks-new/Patient*/ with “_brainMask” suffix
   - Notes: Uses antspynet if available; otherwise falls back to classical ants.get_mask

3) Generate Masked Images
   python 3-generate_masked_images.py
   - Reads images from: 25-nov-registered-new/Patient*/
   - Reads masks from: 26-nov-brain-masks-new-modified/Patient*/ *_brainMask*.nrrd
   - Writes masked images to: 27-nov-brains-extracted-new/Patient*/ with “_brain-extracted” suffix

4) Intra-subject Registration
   python 4-intra_subject_registration.py
   - Registers modalities within the same patient (see file for details/assumptions)

5) Intensity Normalization (Batch)
   python 5-intensity_normalization_batch.py

6) Resample to 1 mm Isotropic
   python 6-resample_to_1mm_isotropic.py
   - Produces outputs similar to 30-nov-Final-MRI-Data-1mm

Helpers
- helpers.py includes:
  - explore_3D_array and explore_3D_array_comparison for quick visual inspection in notebooks
  - explore_3D_array_with_mask_contour for overlaying mask contours
  - add_suffix_to_filename to safely append suffixes to .nii/.nii.gz
  - show_sitk_img_info to print SimpleITK image metadata

Notes and Tips
- GPU is not required; several scripts explicitly disable CUDA to avoid cuDNN mismatches.
- Folder names in the scripts reflect the original workflow dates. You may adapt them to your project, but keep the same structure.
- For antspynet, the first run may download pretrained weights; ensure you have internet connectivity, or rely on the classical fallback.

Troubleshooting
- Missing antspynet/antspyx
  pip install antspyx antspynet
- SimpleITK I/O errors (file not found or format):
  - Verify paths and filename extensions (.nii or .nii.gz)
  - Check that patient folders exist and contain expected files
- Shape mismatch when masking:
  - The masking script will resample the mask to the image grid when possible; if it fails, check image orientations/spaces

License
This project is licensed under the MIT License. See the LICENSE file for details.

Citation
If you use this code in academic work, please cite the repository and the underlying tools (SimpleITK, ANTs/ANTsPy, antspynet).

Acknowledgements
- SimpleITK team
- ANTs/ANTsPy and ANTsX community
- antspynet maintainers
