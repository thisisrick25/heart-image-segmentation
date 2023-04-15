import os
from glob import glob
import shutil
import dicom2nifti
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Make group of dicom files, 57(or 38) slices
def create_groups(in_path, out_path, Number_slices):
    Number_slices = 57
    for folder in tqdm(glob(in_path + '/*'), ncols = 100, desc ="Creating group of dicom files"):
            patient_name = os.path.basename(os.path.normpath(folder))

            # Here we need to calculate the number of folders which mean into how many groups we will divide the number of slices
            number_folders = int(len(glob(folder + '/*')) / Number_slices)

            for i in range(number_folders):
                output_path = os.path.join(out_path, patient_name + '_' + str(i))
                os.mkdir(output_path)

                # Move the slices into a specific folder so that you will save memory in your desk
                for i, file in enumerate(glob(folder + '/*')):
                    if i == Number_slices + 1:
                        break
                    
                    shutil.move(file, output_path)

# convert dicom to nifti
def dcm2nii(in_path, out_path):
    for folder in tqdm(glob(in_path + '/*'), ncols = 100, desc ="Converting dicom to nifti"):
        patient_name = os.path.basename(os.path.normpath(folder))
        dicom2nifti.convert_directory(folder, os.path.join(out_path, patient_name + '.nii.gz'))

# Find empty labels
def fnd_emp(in_path):
    all_folder = []
    for folder in tqdm(glob(os.path.join(in_path, '/*')), ncols = 100, desc ="Finding empty labels"):
        img = nib.load(folder)

        if len(np.unique(img.get_fdata())) > 2:
            print(os.path.basename(os.path.normpath(folder)))
            all_folder.append(os.path.basename(os.path.normpath(folder)))
    
    return all_folder


# create main
if __name__ == '__main__':
    # Create groups of dicom files
    # run the for loop for both images and labels
    in_path_images =  'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/dicom_files/images'
    in_path_labels =  'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/dicom_files/labels'

    out_path_images = 'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/dicom_groups/images'
    out_path_labels = 'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/dicom_groups/labels'

    create_groups(in_path_images, out_path_images, 38)
    create_groups(in_path_labels, out_path_labels, 38)

    # ------------------------------------------------------

    # Convert dicom to nifti
    # run the for loop for both images and labels
    in_path_dcm_grp_images =  'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/dicom_groups/images'
    in_path_dcm_grp_labels =  'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/dicom_groups/labels'
    out_path = 'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/'
    out_path = os.path.join(out_path, 'nifti_files')
    os.mkdir(out_path)

    out_path_nii_images = os.path.join(out_path, 'images')
    out_path_nii_labels = os.path.join(out_path, 'labels')
    os.mkdir(out_path_images)
    os.mkdir(out_path_labels)

    dcm2nii(in_path_dcm_grp_images, out_path_nii_images)
    dcm2nii(in_path_dcm_grp_labels, out_path_nii_labels)
    
    # ------------------------------------------------------

    # Find empty labels
    in_path_nifti_labels = 'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/nifti_files/labels'
    fnd_emp(in_path_nifti_labels)