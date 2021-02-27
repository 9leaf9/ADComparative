# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/18
# version： Python 3.7.8
# @File : mri_extract.py
# @Software: PyCharm

'''提取每个受试者的所有MRI信息'''

import pandas as pd
import os
import shutil
import time
from tqdm import tqdm
from tqdm._tqdm import trange

data = pd.read_csv('/data1/qiaohezhe/ADNI/ADdata/AD_2020_7_31_7_30_2020_processed.csv')
subject_id_list = data['Subject ID']
image_id_list = data['Image Data ID']

# for subject_id in subject_id_list:
#     os.mkdir('/data1/qiaohezhe/ADNI/MRI_ALL/'+subject_id)

filePath = '/data1/qiaohezhe/ADNI/Section1/AD_2020_7_31_10'
file_lsit = os.listdir(filePath)
print(file_lsit)

for index, subject_id in tqdm(enumerate(subject_id_list), total=len(subject_id_list)):
    print("processing {}...".format(subject_id))
    if subject_id in file_lsit:
        dirs = '/data1/qiaohezhe/ADNI/MRI_ALL/' + subject_id_list[index] + '/1/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            '''拷贝至指定目录'''
            target_path = dirs
            source_path = filePath + '/' + subject_id

            if os.path.exists(source_path):
                # root 所指的是当前正在遍历的这个文件夹的本身的地址
                # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
                # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
                for root, dir, files in os.walk(source_path):
                    for file in files:
                        src_file = os.path.join(root, file)
                        shutil.copy(src_file, target_path)
                        # print("{}'s copy is finished", src_file)

        else:
            '''判断指定目录下存在多少个文件'''
            file_number = len(os.listdir('/data1/qiaohezhe/ADNI/MRI_ALL/' + subject_id_list[index]))
            tem_dirs = '/data1/qiaohezhe/ADNI/MRI_ALL/' + subject_id_list[index] + '/' + str(file_number + 1) + '/'
            if not os.path.exists(tem_dirs):
                os.makedirs(tem_dirs)
                source_path = filePath + '/' + subject_id
                target_path = tem_dirs
                if os.path.exists(source_path):
                    for root, dir, files in os.walk(source_path):
                        for file in files:
                            src_file = os.path.join(root, file)
                            shutil.copy(src_file, target_path)
    print("{}'s copy is finshed...".format(subject_id))



