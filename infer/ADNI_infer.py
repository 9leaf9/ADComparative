import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import pandas as pd
from deepbrain import Extractor
from skimage import measure
import itertools
from PIL import Image, ImageEnhance
from skimage.util import random_noise


def all_np(arr):
    # 拼接数组函数
    List = list(itertools.chain.from_iterable(arr))
    arr = np.array(List)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


class AdniDataSet(Dataset):

    def __init__(self, img_path1, img_path2):
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.input_D = 90
        self.input_H = 90
        self.input_W = 90

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, x, y, z])
        new_data = new_data.astype("float32")

        return new_data

    def __scaler__(self, image):
        img_f = image.flatten()
        # find the range of the pixel values
        i_range = img_f[np.argmax(img_f)] - img_f[np.argmin(img_f)]
        # clear the minus pixel values in images
        image = image - img_f[np.argmin(img_f)]
        img_normalized = np.float32(image / i_range)
        return img_normalized

    def get_mri(self):
        img1 = nibabel.load(self.img_path1)
        img2 = nibabel.load(self.img_path2)

        img_array1 = self.__training_data_process__(img1)
        img_array2 = self.__training_data_process__(img2)
        # 2 tensor array
        img_array1 = self.__nii2tensorarray__(img_array1)
        img_array2 = self.__nii2tensorarray__(img_array2)

        return img_array1, img_array2


    def __random_center_crop__(self, data, label):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label > 0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data):
        """
        Random crop with different methods:
        """
        # random center crop
        data = self.__random_center_crop__(data)

        return data

    def __training_data_process__(self, data):
        # crop data according net input size
        data = data.get_data()
        data = self.__drop_invalid_range__(data)
        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        # data = self.__scaler__(data)

        return data

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

