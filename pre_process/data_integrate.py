# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/22
# version： Python 3.7.8
# @File : data_integrate.py
# @Software: PyCharm

'''clincal数据和MRI数据的整合,找到存在最新的MRI的受试者集合，对clincal数据集进行筛选'''

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

subject_list = []
filePath = '/data1/qiaohezhe/MRI_Score/MRI/'
file_lsit = os.listdir(filePath)
for file in file_lsit:
    if os.listdir(filePath + file) and os.path.exists(filePath + file + '/' + 'corr/001_processed.nii'):
            # and os.path.exists(filePath + file + '/' + 'corr/001_processed.nii_predicted_4class_unet3d_extract12_hist15.nii.gz'):
        subject_list.append(file)

ad_data = pd.read_csv('/data1/qiaohezhe/MRI_Score/new_ad_data.csv')
data = ad_data.loc[ad_data['Subject ID'].isin(subject_list)]
data = data.loc[~ad_data['Research Group'].isin(['AD'])]
data.dropna(axis=0, how='any', inplace=True)

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
scaled_score = data[['MMSE Total Score']].apply(max_min_scaler)
data = pd.concat([data, scaled_score], axis=1)
data.to_csv('/data1/qiaohezhe/MRI_Score/new_ad_data_filter_ad.csv', index=False)



