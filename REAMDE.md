# Project planning

## Project description

Artificial intelligence is healthcare imaging. Creating algorithm for automatic heart segmentation.

## Project checklist

- [x] Create/find dataset of heart images

- [x] Preapar the dataset for preprocess
  - [x] if nifti, convert nifti to dicom
  - [x] resize/make group of dicom slices
  - [x] convert the group of dicom slices to nifti
  - [x] check for empty labels and delete them

- [x] Preprocess
  - [x] split dataset into train and test
  - [x] Load the dataset and make dictory of images and labels
  - [x] Normalize/transform the images
  - [x] load the images and labels into dataloader
  - [x] show one image and label

- [x] Train
  - [x] Create model
  - [x] Create loss function
  - [x] Create optimizer
  - [x] Train the model
  - [x] Save the model

- [x] Test
  - [x] Load the model
  - [x] Load the test dataset
  - [x] Test the model
  - [x] Show the results
