# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/24
# version： Python 3.7.8
# @File : file_util.py
# @Software: PyCharm

'''提取受试者最新的MRI 拷贝到新的文件夹'''
import pandas as pd
import os
import nibabel as nib
import shutil
import time
from tqdm import tqdm
from tqdm._tqdm import trange

# data = pd.read_csv('../ADdata/AD_2020_7_31_7_30_2020_processed.csv')
# subject_id_list = data['Subject ID']
# image_id_list = data['Image Data ID']
#
# for subject_id in subject_id_list:
#     os.mkdir('./img/'+subject_id )


# filePath = './img/'
# file_lsit = os.listdir(filePath)
# print(file_lsit)
# for file in file_lsit:
#     souce_path = '../MRI/' + file + '/lastest/001.nii'
#     if os.path.exists(souce_path):
#         lastest_dirs = filePath + file + '/' + 'lastest/'
#         if not os.path.exists(lastest_dirs):
#             os.makedirs(lastest_dirs)
#         shutil.copy(souce_path, './img/' + file + '/lastest/001.nii')
#         print("{} copy is finished" .format(file))

'''查找不存在 lastest nii的文件'''
# filePath = '../MRI/'
# file_lsit = os.listdir(filePath)
# for file in file_lsit:
#     if not os.path.exists(filePath + file + '/lastest/001.nii'):
#         print(file)

'''查找有无nii大小异常的文件 MRI大小不为3'''
# filePath = '../MRI/'
# file_lsit = os.listdir(filePath)
# for file in file_lsit:
#     if os.path.exists(filePath + file + '/lastest/001.nii'):
#         img = nib.load(filePath + file + '/lastest/001.nii')
#         # 打印文件信息
#         # print(img)
#         # print(len(img.dataobj.shape))
#         if len(img.dataobj.shape) != 3:
#             print(file)

'''查看 005.nii 存在的受试者个数'''
# filePath = '/data1/qiaohezhe/ADNI/MRI_lastest/'
# file_lsit = os.listdir(filePath)
# num = 0
# for file in file_lsit:
#     if os.path.exists(filePath + file + '/lastest/001.nii_predicted_4class_unet3d_extract12_hist15.nii.gz'):
#         img = nib.load(filePath + file + '/lastest/001.nii_predicted_4class_unet3d_extract12_hist15.nii.gz')
#         # 打印文件信息
#         # print(img)
#         # print(len(img.dataobj.shape))
#         if len(img.dataobj.shape) != 3:
#             print(file)
#         num += 1
# print("the number of subjects who exist 005.nii is {}".format(num))
import numpy as np
import itertools
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
    # {1: 2, 2: 3, 3: 3, 4: 1, 5: 3}
'''查看是否有全0的mask'''

filePath = '/data1/qiaohezhe/ADNI/MRI_corr_all/'
file_lsit = os.listdir(filePath)
num = 0
for file in file_lsit:
    if os.path.exists(filePath + file + '/corr/001_processed.nii_predicted_4class_unet3d_extract12_hist15.nii.gz'):
        mask = nib.load(filePath + file + '/corr/001_processed.nii_predicted_4class_unet3d_extract12_hist15.nii.gz')
        img = nib.load(filePath + file + '/corr/001_processed.nii')
        mask = mask.get_data()
        values = [v for v in all_np(mask).keys()]
        mask[mask == values[1]] = 0
        mask[mask == values[2]] = 0
        mask[mask == values[3]] = 1
        img = img.get_data()
        volume = mask*img
        print(file)
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        # if data.std() == 0:
        #     print(file)