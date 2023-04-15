# code to find the optimum number of slices to make group of dicom files

# count number of files in each folders in a directory
import os
from glob import glob
import numpy as np

path = 'C:/Users/swapn/code/AI Healthcare Imaging/datasets/Task02_Heart/dicom_files/images'
count = []

for folder in glob(path + '/*'):
    count.append(len(os.listdir(folder)))

print(count)

# find avarage ceil value of count
n = np.ceil(np.mean(count))
print(n)

# make group of two folders for each patient
print(n/2)

# #make group of three folders for each patient
# print(n/3)
