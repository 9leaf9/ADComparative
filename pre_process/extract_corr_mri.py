# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/9/26 
# version： Python 3.7.8
# @File : extract_corr_mri.py
# @Software: PyCharm

'''对每个受试者提取对应的MRI， 并拷贝到相应的文件夹'''

import os
import pandas as pd
import shutil
from tqdm import tqdm

# lastest_data = pd.read_csv('/data1/qiaohezhe/ADNI/ADdata/AD_2020_7_31_7_30_2020_latest.csv')
# img_ids = lastest_data['Image Data ID']
# subject_ids = lastest_data['Subject ID']
#
# '''对每个受试者提取相应的MRI, 并拷贝到相应的文件夹'''
#
# filePath = '/data1/qiaohezhe/ADNI/MRI_ALL/'
# targetPath = '/data1/qiaohezhe/ADNI/MRI_corr/'
#
# sum = 0
# for index, file in tqdm(enumerate(subject_ids), total=len(subject_ids)):
#     if os.path.exists(filePath + file):
#         print("processing {}...".format(file))
#         lastest_dirs = targetPath + file + '/' + 'corr/'
#         if not os.path.exists(lastest_dirs):
#             os.makedirs(lastest_dirs)
#         source_file = filePath + file + '/'
#         mri_path = ""
#         for root, dir, files in os.walk(source_file):
#             for temp_file in files:
#                 mri_file = os.path.join(root, temp_file)
#                 if mri_file.find('Mask') == -1 and mri_file.find(str(img_ids[index])) != -1 and mri_file.find('MR_MT1') != -1 :
#                     mri_path = mri_file
#
#         print(mri_path)
#         '''文件拷贝'''
#         if mri_path != "":
#             shutil.copy(mri_path, lastest_dirs + '/001.nii')
#             print("{}'s lastest copy is finshed...".format(file))
#             sum +=1
#         else:
#             print("The lastest MRI in {} is not exist".format(file))
#
#     else:
#         print("{} is empty".format(file))
#
# print(sum)


data = pd.read_csv('/data1/qiaohezhe/MRI_Score/idaSearch_12_01_2020.csv')
print(data)
img_ids = data['Image ID']
subject_ids = data['Subject ID']
new_image_ids = []

'''对每个受试者提取相应的MRI, 并拷贝到相应的文件夹'''

filePath = '/data1/qiaohezhe/ADNI/MRI_ALL/'
targetPath = '/data1/qiaohezhe/MRI_Score/MRI/'

sum = 0
for index, file in tqdm(enumerate(subject_ids), total=len(subject_ids)):
    if os.path.exists(filePath + file):
        print("processing {}...".format(file))
        lastest_dirs = targetPath + file + '/' + 'corr/'
        if not os.path.exists(lastest_dirs):
            os.makedirs(lastest_dirs)
        if not os.path.exists(lastest_dirs + '/001.nii'):
            source_file = filePath + file + '/'
            mri_path = ""
            for root, dir, files in os.walk(source_file):
                for temp_file in files:
                    mri_file = os.path.join(root, temp_file)
                    if mri_file.find('Mask') == -1 and mri_file.find(str(img_ids[index])) != -1:
                        mri_path = mri_file
                    # if mri_file.find('Mask') == -1 and mri_file.find(str(img_ids[index])) != -1 and (mri_file.find(
                    #         'MR_MPR') != -1 or mri_file.find('MR_MT1') != -1):

            print(mri_path)
            '''文件拷贝'''
            if mri_path != "":
                shutil.copy(mri_path, lastest_dirs + '/001.nii')
                print("{}'s corr MRI copy is finshed...".format(file))
                sum += 1
                new_image_ids.append(img_ids[index])
            else:
                print("The corr MRI in {} is not exist".format(file))

    else:
        print("{} is empty".format(img_ids[index]))

print(sum)
new_data = data[data['Image ID'].isin(new_image_ids)]
new_data.to_csv('/data1/qiaohezhe/MRI_Score/new_ad_data.csv')