# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/26
# version： Python 3.7.8
# @File : mri_util2.py
# @Software: PyCharm


import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from os import getcwd, chdir, listdir, remove, walk
from sklearn.model_selection import train_test_split
# import defined functions for preprocessing
from util.mri_function import *
from util.image_util import *
###############################################  Supplementary code ###########################################
# selected a random image to check the outcome of each preprocessing_org steps
# icheck = np.load(filtered + "/" + listdir(filtered)[16])
# check skull-stripping
example_filename = '../MRI/002_S_0619/lastest/001.nii'
img = nib.load(example_filename)
iskull = skull_stripper(np.array(img.get_data()))
# view in 2D plot
# plt.imshow(iskull[128], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
# multislice viewer
# multi_slice_viewer(iskull, [0, 400, 0, 400])
new_image = nib.Nifti1Image(iskull, np.eye(4))
showNii(new_image)


# save image to view in 3D space
# img_skull = nib.Nifti1Image(iskull, affine=np.eye(4))
# img_skull.header.get_xyzt_units()
# img_skull.to_filename(os.path.join('D:/ADNI_unzipped/view', 'img_skull.nii'))

# check brain cropping
icrop = crop(iskull)
# plt.imshow(icrop[80], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
# multislice viewer
# multi_slice_viewer(icrop, [0, 400, 0, 400])
new_image = nib.Nifti1Image(iskull, np.eye(4))
showNii(new_image)


# save image to view in 3D space
# img_crop = nib.Nifti1Image(icrop, affine=np.eye(4))
# img_crop.header.get_xyzt_units()
# img_crop.to_filename(os.path.join('D:/ADNI_unzipped/view', 'img_crop.nii'))

# check resizing
istd = resizer(icrop)
# plt.imshow(icrop[30], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
# multislice viewer
# multi_slice_viewer(istd, [0, 400, 0, 400])
new_image = nib.Nifti1Image(iskull, np.eye(4))
showNii(new_image)



# save image to view in 3D space
# img_std = nib.Nifti1Image(istd, affine=np.eye(4))
# img_std.header.get_xyzt_units()
# img_std.to_filename(os.path.join('D:/ADNI_unzipped/view', 'img_std.nii'))


# checking intensity-normalization/the final products
inor = scaler(istd)
# plt.imshow(inor[30], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
# multislice viewer
# multi_slice_viewer(inor, [0, 400, 0, 400])
# save image to view in 3D space
new_image = nib.Nifti1Image(iskull, np.eye(4))
showNii(new_image)

# img_nor = nib.Nifti1Image(inor, affine=np.eye(4))
# img_nor.header.get_xyzt_units()
# img_nor.to_filename(os.path.join('D:/ADNI_unzipped/view', 'img_nor.nii'))

# %% remove the images failed to pass the preprocessing_org
# loaded the images that failed with the preprocessing (error message) - image 80
# f_img = np.load(processed + "/" + listdir(processed)[80])
# # double check the quality of the failed image in 2D and 3D view
# plt.imshow(f_img[128], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
# multi_slice_viewer(f_img, [0, 600, 0, 600])
# # remove the the image
# remove(processed + "/" + listdir(processed)[80])

# %%
# Check the quality of each data augmentation technique

# affine atransformation is processed after skull-stripping before brain cropping
img = skull_stripper(inor)
print(img.shape)
# plt.imshow(img[120], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
multi_slice_viewer(img, [0, 600, 0, 600])

"""rotation"""
# rotate a slice of a MRI image
from scipy.ndimage.interpolation import rotate

plt.imshow(img[120, :, :], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
img_rot2d = rotate(img, -15, mode='nearest', axes=(0, 1), reshape=False)
plt.imshow(img[120], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

# check the rotated 3D image
img_rot = random_rotation_3d(img)
plt.imshow(img_rot[120], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_rot[:, 110, :], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_rot[:, :, 80], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

multi_slice_viewer(img, [0, 600, 0, 600])
multi_slice_viewer(img_rot, [0, 600, 0, 600])

"""flip"""
# check flipped 3D image
img_flip = flip(img)
plt.imshow(img[120], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_flip[121], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

multi_slice_viewer(img, [0, 600, 0, 600])
multi_slice_viewer(img_flip, [0, 600, 0, 600])

"""elastic transformation"""
# check a slice of MRI images after elastic transformation
plt.imshow(img[120], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
slice_ela = elastic_transform(img[120])
plt.imshow(slice_ela, aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

# check a 3D image after elastic transformation
img_ela = elastic_3d(img)
plt.imshow(img_ela[120], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_ela[:, 110, :], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_ela[:, :, 80], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

multi_slice_viewer(img, [0, 600, 0, 600])
multi_slice_viewer(img_ela, [0, 600, 0, 600])

"""contrast and sharpness"""
# pixel-level transformation are processed after intensity normalization
preprocessed = "D:/ADNI_unzipped/processed"
img2 = np.load(preprocessed + '/' + listdir(preprocessed)[571])
print(img2.shape)
# plt.imshow(img2[32], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
multi_slice_viewer(img2, [0, 600, 0, 600])

# check a slice of MRI image with changing contrast and sharpness
img_sharp = sharp(img2[32], 1.98)
img_contra = contra(img2[32], 1.89)
plt.imshow(img2[32], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_sharp, aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_contra, aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

# check a 3D image after chaing contrast and sharpness
img_con3d = contra_3d(img2)
img_sa3d = sharp_3d(img2)
plt.imshow(img_con3d[:, :, 28], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_sa3d[:, 30, :], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

multi_slice_viewer(img2, [0, 600, 0, 600])
multi_slice_viewer(img_con3d, [0, 600, 0, 600])
multi_slice_viewer(img_sa3d, [0, 600, 0, 600])

"""noise"""
# checking adding noise to a 2D slice
from skimage.util import random_noise

img_noisy = random_noise(img2[32], mode='speckle', seed=None)
plt.imshow(img_noisy, aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

# check a #D image after adding in noise
img_no = noise(img2)
plt.imshow(img_no[:, 25, :], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
plt.imshow(img_no[:, :, 27], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")

multi_slice_viewer(img2, [0, 600, 0, 600])
multi_slice_viewer(img_no, [0, 600, 0, 600])
