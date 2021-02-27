# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/24
# version： Python 3.7.8
# @File : file_copy.py
# @Software: PyCharm

'''提取受试者最新的MRI 拷贝到新的文件夹'''
import pandas as pd
import os
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


filePath = './img/'
file_lsit = os.listdir(filePath)
print(file_lsit)
for file in file_lsit:
    souce_path = '../MRI/' + file + '/lastest/001.nii'
    if os.path.exists(souce_path):
        lastest_dirs = filePath + file + '/' + 'lastest/'
        if not os.path.exists(lastest_dirs):
            os.makedirs(lastest_dirs)
        shutil.copy(souce_path, './img/' + file + '/lastest/001.nii')
        print("{} copy is finished" .format(file))