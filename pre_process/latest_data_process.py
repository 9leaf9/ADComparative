# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/21
# version： Python 3.7.8
# @File : latest_data_process.py
# @Software: PyCharm
'''处理的得到的每个受试者最新的数据  拆分+归一化'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# """load and store the subject_id and labels for further data splitting"""
# # loading the labels
# labels = pd.read_csv('./ADdata/AD_2020_7_31_7_30_2020.csv')
#
# # find the unique subject-ids
# uq_ids = set(labels['Subject'])
#
# # define a dictionary to store subject_id(keys) and class labels(values)
# sub_labels = dict()
# for id in uq_ids:
#     if id not in sub_labels.keys():
#         label = ''.join(np.unique(labels['Group'][labels['Subject'] == id]))
#         sub_labels[id] = label
#
# # %%
# """ split the subjects (subject id) into training, validation and testing"""
# subs = [*sub_labels]
# labs = [sub_labels[sub] for sub in subs]

ad_data = pd.read_csv('/data1/qiaohezhe/ADNI/ADdata/AD_2020_7_31_7_30_2020_latest.csv')
ad_cols = ["Image Data ID", "Subject ID", "Group", "Sex", "Age", "Visit", "APOE A1", "APOE A2", "MMSE Total Score", "GDSCALE Total Score", "Global CDR", "FAQ Total Score", "NPI-Q Total Score"]
data = ad_data[ad_cols]

scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
'''age score'''
scaled_column = scaler.fit_transform(data[['Age']])

data = pd.concat([data, pd.DataFrame(scaled_column, columns=['scaled_Age'])], axis=1)
scaled_column = scaler.fit_transform(data[['APOE A1']])
data = pd.concat([data, pd.DataFrame(scaled_column, columns=['scaled_APOE A1'])], axis=1)

scaled_column = scaler.fit_transform(data[['APOE A2']])
data = pd.concat([data, pd.DataFrame(scaled_column, columns=['scaled_APOE A2'])], axis=1)

scaled_column = scaler.fit_transform(data[['MMSE Total Score']])
data = pd.concat([data, pd.DataFrame(scaled_column, columns=[['scaled_MMSE Total Score']])], axis=1)

scaled_column = scaler.fit_transform(data[['GDSCALE Total Score']])
data = pd.concat([data, pd.DataFrame(scaled_column, columns=['scaled_GDSCALE Total Score'])], axis=1)

scaled_column = scaler.fit_transform(data[['Global CDR']])
data = pd.concat([data, pd.DataFrame(scaled_column, columns=['scaled_Global CDR'])], axis=1)

scaled_column = scaler.fit_transform(data[['FAQ Total Score']])
data = pd.concat([data, pd.DataFrame(scaled_column, columns=['scaled_Global CDR'])], axis=1)

scaled_column = scaler.fit_transform(data[['NPI-Q Total Score']])
data = pd.concat([data, pd.DataFrame(scaled_column, columns=['scaled_NPI-Q Total Score'])], axis=1)

'''sex'''
data["count"] = 1
gender = data.pivot(index='Subject ID', columns="Sex", values="count")
gender.reset_index(inplace=True)
gender.fillna(0, inplace=True)
data = data.merge(gender, on=['Subject ID'], how='left')


'''MCI AD NC'''
group = data.pivot(index='Subject ID', columns="Group", values="count")
group.reset_index(inplace=True)
group.fillna(0, inplace=True)
data = data.merge(group, on=['Subject ID'], how='left')

data.to_csv('/data1/qiaohezhe/ADNI/ADdata/AD_2020_7_31_7_30_2020_lastest_processed.csv', index=False)