# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/10/17 
# version： Python 3.7.8
# @File : data_spilt_argument.py
# @Software: PyCharm
import sys

sys.path.append('/home/qiaohezhe/code/MRI_AD/util')
import os
# print(os.getcwd())
from deepbrain import Extractor
from image_util import *
from tqdm import tqdm
import nibabel as nb
import pandas as pd
import random
from random import shuffle
from PIL import Image, ImageEnhance
from skimage.transform import resize
from skimage.util import random_noise
from skimage.measure import label, regionprops
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from os import listdir, walk, remove
import re
import random
import shutil
import numpy as np
import nibabel as nib
from random import uniform
from deepbrain import Extractor

def skull_stripper(image):
    ext = Extractor()
    prob = ext.run(image)
    # set the threshold for extractor to extract the brain
    mask = prob < 0.7
    img_filtered = np.ma.masked_array(image, mask=mask)
    # fill the background to 0
    img_filtered = img_filtered.filled(0)

    return img_filtered


# 3d image rotation
def random_rotation_3d(img):
    """ Randomly rotate an image by a random angle (-15, 15).

    Returns:
    a rotated 3D image
    """
    max_angle = 20

    if bool(random.getrandbits(1)):
        # rotate along z-axis
        angle = uniform(-max_angle, max_angle)
        img1 = rotate(img, angle, mode='nearest', axes=(0, 1), order=1, reshape=False)
        # print(angle)
        # print("Z-axis")
    else:
        img1 = img

    # rotate along y-axis
    if bool(random.getrandbits(1)):
        angle = uniform(-max_angle, max_angle)
        img2 = rotate(img1, angle, mode='nearest', axes=(0, 2), order=1, reshape=False)
        # print(angle)
        # print("Y-axis")
    else:
        img2 = img1

    # rotate along x-axis
    if bool(random.getrandbits(1)):
        angle = uniform(-max_angle, max_angle)
        img3 = rotate(img2, angle, mode='nearest', axes=(1, 2), order=1, reshape=False)
        img3 = np.float32(img3)
        # print(angle)
        # print("X-axis")
    else:
        img3 = np.float32(img2)

    return img3


# 3d image flipping
def flip(img):
    axis = random.sample(range(0, 2), 1)[0]
    new_img = np.zeros(img.shape)
    # print(axis)
    if axis == 0:
        for i in range(img.shape[0]):
            new_img[i] = np.fliplr(img[i])
    if axis == 1:
        for i in range(img.shape[1]):
            new_img[:, i, :] = np.fliplr(img[:, i, :])
    # no flip on the Z-axis, as the brain will turn upside down, which is not possible for a normal brain
    new_img = np.float32(new_img)

    return new_img


# elastic transofmraiton on 2d slice
def elastic_transform(image, alpha=80, sigma=10, random_state=None, sd=29):
    """
    alpha = scaling factor the deformation;
    sigma = smooting factor

    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    random_state.seed(sd)
    # print(random_state.rand(*shape))

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


# elastic transoformation on 3d images
def elastic_3d(img):
    """"
    using elastic_transform  to transform all slices of one dimension
    seed is fixed per dimension
    transformation happens at one of the three dimensions
    """
    # import random
    axis = random.sample(range(0, 3), 1)[0]
    new_img = np.zeros(img.shape)
    # print(axis)
    if axis == 0:
        # generate a seed for one type of elastic transformation for all images along this axis
        rand = random.sample(range(0, 100), 1)[0]
        for i in range(img.shape[0]):
            new_img[i] = elastic_transform(img[i], sd=rand)
    if axis == 1:
        rand = random.sample(range(0, 100), 1)[0]
        for i in range(img.shape[1]):
            new_img[:, i, :] = elastic_transform(img[:, i, :], sd=rand)
    if axis == 2:
        rand = random.sample(range(0, 100), 1)[0]
        for i in range(img.shape[2]):
            new_img[:, :, i] = elastic_transform(img[:, :, i], sd=rand)
    # print(rand)
    new_img = np.float32(new_img)

    return new_img


# chaing image cotrast for 2d slice
def contra(img, factor):
    # change the pixel values back to 0 - 255
    img = img * 255
    img = Image.fromarray(img)
    img = img.convert('L')

    # change contrast
    img1 = ImageEnhance.Contrast(img)
    img_con = img1.enhance(factor)

    # converting back to np array
    img_final = np.float32(np.asarray(img_con) / 255)
    return img_final


# changing image contrast for 3d images
def contra_3d(img):
    new_img = np.zeros(img.shape)
    """creating contrast factor 
    factors ranging between 0.7 - 1.5 will make the augmented image close to original image"""
    factor = 1
    while factor < 1.4 and factor > 0.7:
        factor = np.random.uniform(0.4, 1.9)
    # print(factor)
    # go through each slice to change the contrast and sharpness
    for i in range(img.shape[0]):
        new_img[i] = contra(img[i], factor)

    return np.float32(new_img)


# changing image sharpness for 2d slice
def sharp(img, factor):
    # change the pixel values back to 0 - 255
    img = img * 255
    img = Image.fromarray(img)
    img = img.convert('L')

    # change sharpness
    img1 = ImageEnhance.Sharpness(img)
    img_sharp = img1.enhance(factor)

    img_final = np.float32(np.asarray(img_sharp) / 255)
    return img_final


# changing image sharpness for 3d images
def sharp_3d(img):
    new_img = np.zeros(img.shape)
    """creating contrast factor 
    factors ranging between 0.7 - 1.5 will make the augmented image close to original image"""
    factor = 1
    while factor < 1.5 and factor > 0.7:
        factor = np.random.uniform(0.4, 2)
    # print(factor)

    for i in range(img.shape[0]):
        new_img[i] = sharp(img[i], factor)

    return np.float32(new_img)


# adding in noise to 3d images
def noise(img):
    new_img = np.zeros(img.shape)
    # choose one type of noise to add to the image
    mode = ['gaussian', 's&p', 'poisson', 'speckle']
    random.shuffle(mode)
    idx = random.sample(range(0, 4), 1)[0]
    noise = mode[idx]
    # print(noise)

    # fix a seed for all slices to be added with the exact same noise
    sd = random.sample(range(0, 100), 1)[0]

    new_img = random_noise(img, mode=noise, seed=sd)

    return np.float32(new_img)

def scaler(image):
    img_f = image.flatten()
    # find the range of the pixel values
    i_range = img_f[np.argmax(img_f)] - img_f[np.argmin(img_f)]
    # clear the minus pixel values in images
    image = image - img_f[np.argmin(img_f)]
    img_normalized = np.float32(image / i_range)
    # print(M_normalized.shape)
    return img_normalized


if __name__ == "__main__":
    file_path = '/data1/qiaohezhe/AD2/MCI/'
    train_save_path = '/data1/qiaohezhe/AD2/mci_nc/'
    # val_save_path = "/data1/qiaohezhe/MRI_Score/mci_nc/val/"
    data_path = '/data1/qiaohezhe/AD2/mci_data.csv'
    data = pd.read_csv(data_path)
    subject_lsit = data['Subject ID'].tolist()
    group_lsit = data['Research Group'].tolist()
    score_list = data['MMSE Total Score'].tolist()
    random_index = [i for i in range(len(subject_lsit))]
    shuffle(random_index)
    train_index = random_index
    # val_index = random_index[int(len(subject_lsit)*0.75):]

    '''Train'''
    number = 146
    labels = []
    for index in tqdm(train_index):
        mri_path = file_path + subject_lsit[index] + '/corr/001_processed.nii'
        if os.path.exists(mri_path):
            '''1.读取  2.增强   3.保存'''
            img = nib.load(mri_path)
            nb.save(img, train_save_path + "{}.nii".format(str(number)))
            labels.append([number, group_lsit[index], score_list[index]])
            number += 1

            # img_data = img.get_data()
            #
            # roa_img_data = random_rotation_3d(img_data)
            # roa_img = nib.Nifti1Image(roa_img_data, np.eye(4))
            # nb.save(roa_img, train_save_path + "{}.nii".format(str(number)))
            # labels.append([number, group_lsit[index], score_list[index]])
            # number += 1
            #
            # scaler_img_data = scaler(img_data)
            # ela_img_data = elastic_3d(scaler_img_data)
            # ela_img_data = ela_img_data * 255.0
            # ela_img = nib.Nifti1Image(ela_img_data, np.eye(4))
            # nb.save(ela_img, train_save_path + "{}.nii".format(str(number)))
            # labels.append([number, group_lsit[index], score_list[index]])
            # number += 1
            #
            # scaler_img_data = scaler(img_data)
            # sharp_img_data = sharp_3d(scaler_img_data)
            # sharp_img_data = sharp_img_data * 255.0
            # sharp_img = nib.Nifti1Image(sharp_img_data, np.eye(4))
            # nb.save(sharp_img, train_save_path + "{}.nii".format(str(number)))
            # labels.append([number, group_lsit[index], score_list[index]])
            # number += 1
            #
            # scaler_img_data = scaler(img_data)
            # cont_img_data = contra_3d(scaler_img_data)
            # cont_img_data =  cont_img_data * 255.0
            # cont_img = nib.Nifti1Image(cont_img_data, np.eye(4))
            # nb.save(cont_img, train_save_path + "{}.nii".format(str(number)))
            # labels.append([number, group_lsit[index],score_list[index]])
            # number += 1

            # scaler_img_data = scaler(img_data)
            # noise_img_data = noise(scaler_img_data)
            # noise_img_data = noise_img_data * 255.0
            # noise_img = nib.Nifti1Image(noise_img_data, np.eye(4))
            # nb.save(noise_img, train_save_path + "{}.nii".format(str(number)))
            # labels.append([number, group_lsit[index], score_list[index]])
            # number += 1

    name = ['index', 'group', 'score']
    data_csv = pd.DataFrame(columns=name, data=labels)
    data_csv.to_csv("/data1/qiaohezhe/AD2/adni2_mci_nc_val_label2.csv", index=False)

    # '''Valid'''
    # labels = []
    # number = 0
    # for index in tqdm(val_index):
    #     mri_path = file_path + subject_lsit[index] + '/corr/001_processed.nii'
    #     if os.path.exists(mri_path):
    #         '''1.读取  2.增强   3.保存'''
    #         img = nib.load(mri_path)
    #         nb.save(img, val_save_path + "{}.nii".format(str(number)))
    #         labels.append([number, group_lsit[index], score_list[index]])
    #         number += 1
    #
    # name = ['index', 'group', 'score']
    # data_csv = pd.DataFrame(columns=name, data=labels)
    # data_csv.to_csv("/data1/qiaohezhe/MRI_Score/mci_nc/val_label.csv", index=False)