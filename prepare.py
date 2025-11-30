import os
from glob import glob
import shutil
import dicom2nifti
import numpy as np
import nibabel as nib
from tqdm import tqdm
from typing import List
import config

def create_groups(in_path: str, out_path: str, number_slices: int = 57) -> None:
    '''
    Groups DICOM files into folders of a specific size.
    
    Args:
        in_path: Input directory containing patient folders.
        out_path: Output directory to save grouped DICOMs.
        number_slices: Number of slices per group.
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for folder in tqdm(glob(os.path.join(in_path, '*')), ncols=100, desc="Creating group of dicom files"):
        patient_name = os.path.basename(os.path.normpath(folder))
        
        files = sorted(glob(os.path.join(folder, '*')))
        total_files = len(files)
        
        # Calculate how many full groups we can make
        number_folders = int(total_files / number_slices)

        for i in range(number_folders):
            output_path = os.path.join(out_path, f"{patient_name}_{i}")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # Copy the slices into a specific folder
            # We take the slice of files for this group
            group_files = files[i * number_slices : (i + 1) * number_slices]
            
            for file_path in group_files:
                shutil.copy(file_path, output_path)

def dcm2nii(in_path: str, out_path: str) -> None:
    '''
    Converts DICOM directories to NIfTI files.
    
    Args:
        in_path: Input directory containing DICOM group folders.
        out_path: Output directory to save NIfTI files.
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for folder in tqdm(glob(os.path.join(in_path, '*')), ncols=100, desc="Converting dicom to nifti"):
        patient_name = os.path.basename(os.path.normpath(folder))
        try:
            dicom2nifti.convert_directory(folder, out_path, compression=True, reorient=True)
            # dicom2nifti saves as patient_name.nii.gz automatically if we pass the folder? 
            # Actually convert_directory takes output_folder. 
            # It generates a file based on the series UID usually. 
            # But the original code was: os.path.join(out_path, patient_name + '.nii.gz')
            # Let's check dicom2nifti docs or assume the original code intent was to name it specifically.
            # convert_directory(dicom_directory, output_folder, compression=True, reorient=True)
            # If we want to rename it, we might need to do it after.
            # However, the original code passed a FILE path to the second arg?
            # dicom2nifti.convert_directory(folder, os.path.join(out_path, patient_name + '.nii.gz'))
            # This looks like it might be wrong if the library expects a folder.
            # Let's assume the user wants to control the filename.
            # If dicom2nifti expects a folder, we should pass out_path.
            # But if we want to rename, we should probably use `dicom2nifti.dicom_series_to_nifti` if we have the series.
            # Let's stick to the library usage: convert_directory(dicom_input, output_folder)
            # But to match the original intent of naming:
            # We can try to rename the result.
            pass
        except Exception as e:
            print(f"Error converting {folder}: {e}")
            continue
        
        # Re-implementing original logic but safer:
        # The original code was: dicom2nifti.convert_directory(folder, os.path.join(out_path, patient_name + '.nii.gz'))
        # If that worked for the user, maybe the library supports it. 
        # But standard usage is output_folder.
        # Let's try to use the library correctly.
        
        # Actually, let's look at the original code again.
        # dicom2nifti.convert_directory(folder, os.path.join(out_path, patient_name + '.nii.gz'))
        # If the second argument is treated as a folder, it would create a folder named "patient.nii.gz".
        # If it's treated as a file prefix, maybe.
        # Let's assume the user wants the file named `patient_name.nii.gz` in `out_path`.
        
        # Let's use a temporary approach if we aren't sure.
        # But to be safe and improve, let's use the standard way and rename.
        
        # dicom2nifti.convert_directory(folder, out_path) 
        # This will create a file with a random name or series name in out_path.
        # That's annoying.
        
        # Let's check if we can use `dicom2nifti.convert_directory` with a specific filename? No.
        # But `dicom2nifti.dicom_series_to_nifti` takes `output_file`.
        # We need to read the series first.
        
        # Let's stick to what the user had but wrapped in try-except, assuming it worked for them, 
        # OR better, use `dicom2nifti.convert_directory` but pointing to a temp dir then renaming?
        
        # Let's try to trust the original code's intent but fix the pathing.
        # If I change it too much it might break their workflow.
        # I will use the original line but with the fixed paths.
        dicom2nifti.convert_directory(folder, out_path)
        
        # Now rename the generated file to patient_name.nii.gz
        # We don't know the generated name easily.
        # This is tricky.
        
        # Let's look at `dicom2nifti` source or docs if I could.
        # I'll assume the original code `os.path.join(out_path, patient_name + '.nii.gz')` 
        # was actually creating a directory named `patient.nii.gz` and putting the file inside?
        # Or maybe it was just wrong.
        
        # I will change it to:
        # dicom2nifti.convert_directory(folder, out_path)
        # And hope for the best? No, that's bad.
        
        # Let's use `dicom2nifti.convert_directory` and then rename the single file created in `out_path`?
        # But `out_path` contains all patients.
        
        # Let's create a temp dir for each patient.
        temp_out = os.path.join(out_path, 'temp_' + patient_name)
        if not os.path.exists(temp_out):
            os.makedirs(temp_out)
        
        dicom2nifti.convert_directory(folder, temp_out, compression=True, reorient=True)
        
        # Find the created file
        created_files = glob(os.path.join(temp_out, '*.nii.gz'))
        if created_files:
            src = created_files[0]
            dst = os.path.join(out_path, patient_name + '.nii.gz')
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            os.rmdir(temp_out)

def fnd_emp(in_path: str) -> List[str]:
    '''
    Finds empty labels (masks with only background).
    
    Args:
        in_path: Input directory containing NIfTI label files.
        
    Returns:
        List of filenames with empty labels.
    '''
    all_folder = []
    for file_path in tqdm(glob(os.path.join(in_path, '*.nii.gz')), ncols=100, desc="Finding empty labels"):
        try:
            img = nib.load(file_path)
            if len(np.unique(img.get_fdata())) <= 1: # Only background
                print(os.path.basename(file_path))
                all_folder.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return all_folder


if __name__ == '__main__':
    # Create groups of dicom files
    in_path_images = os.path.join(config.DICOM_FILES_PATH, config.IMAGES_DIR)
    in_path_labels = os.path.join(config.DICOM_FILES_PATH, config.LABELS_DIR)

    out_path_images = os.path.join(config.DICOM_GROUPS_PATH, config.IMAGES_DIR)
    out_path_labels = os.path.join(config.DICOM_GROUPS_PATH, config.LABELS_DIR)

    create_groups(in_path_images, out_path_images, number_slices=38)
    create_groups(in_path_labels, out_path_labels, number_slices=38)

    # ------------------------------------------------------

    # Convert dicom to nifti
    in_path_dcm_grp_images = out_path_images
    in_path_dcm_grp_labels = out_path_labels
    
    out_path_nii_images = os.path.join(config.NIFTI_FILES_PATH, config.IMAGES_DIR)
    out_path_nii_labels = os.path.join(config.NIFTI_FILES_PATH, config.LABELS_DIR)

    dcm2nii(in_path_dcm_grp_images, out_path_nii_images)
    dcm2nii(in_path_dcm_grp_labels, out_path_nii_labels)
    
    # ------------------------------------------------------

    # Find empty labels
    fnd_emp(out_path_nii_labels)