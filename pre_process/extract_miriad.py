# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/9/26 
# version： Python 3.7.8
# @File : extract_miriad.py
# @Software: PyCharm

'''处理MRIIAD数据'''
import os
import pandas as pd
import shutil
from tqdm import tqdm

#
# source_file = '/data1/miriad/'
# targetPath = '/data1/qiaohezhe/miriad/mri/'
# total = 0
# for root, dir, files in os.walk(source_file):
#     for temp_file in files:
#         mri_file = os.path.join(root, temp_file)
#         if mri_file != "":
#             shutil.copy(mri_file, targetPath + temp_file)
#             print("{}'s corr MRI copy is finshed...".format(targetPath + temp_file))
#             total += 1
#         else:
#             print("The corr MRI in {} is not exist".format(file))
#
#
# print("total", total)


'''根据MIMRI 来提取MRI'''
source_path = "/data1/qiaohezhe/miriad/mri_processed/"
target_path = "/data1/qiaohezhe/miriad/ad_vs_nc/"
data = pd.read_csv("/data1/qiaohezhe/miriad/blackcl_1_5_2021_2_34_13.csv")
subjects = data['Label']
scores = data['MMSE']

labels = []
number = 0
for sub, score in zip(subjects, scores):
    sub_id = sub.split("_")[1]
    mri_id = sub.split("_")[2]
    mri_list = os.listdir(source_path)
    for mri in mri_list:
        if sub_id == mri.split("_")[1] and int(mri_id) == int(mri.split("_")[4]):
            if mri.split("_")[6] == "1.nii":
                shutil.copy(source_path + mri, target_path + "{}.nii".format(str(number)))
                if mri.split("_")[2] == "AD":
                    labels.append([number, "AD", score])
                else:
                    labels.append([number, "NC", score])
    number += 1


name = ['index', 'group', 'score']
data_csv = pd.DataFrame(columns=name, data=labels)
data_csv.to_csv("/data1/qiaohezhe/miriad/miriad_val_label.csv", index=False)

