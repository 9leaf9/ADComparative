# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/20
# version： Python 3.7.8
# @File : extract_info.py
# @Software: PyCharm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ad_data = pd.read_csv('/data1/qiaohezhe/ADNI/ADdata/AD_lastest_filter_all_MR_MPR_processed.csv')

'''统计受试者个数及访问次数'''
# ad_cols = ["Image Data ID", "Subject ID", "Group", "Sex", "Age", "Visit", "Acq Date"]
# ad_data = ad_data[ad_cols]
# count = ad_data.loc[:, "Subject ID"].value_counts()
# print(count.values)
# clinical_data = pd.read_csv('./ADdata/idaSearch_7_30_2020.csv')
# clinical_cols = ["Subject ID", "APOE A1", "APOE A2", "MMSE Total Score", "GDSCALE Total Score", "Global CDR", "FAQ Total Score", "NPI-Q Total Score"]
# clinical_data = clinical_data[clinical_cols]
# data = pd.merge(ad_data, clinical_data, on=['Subject ID'], how='left')

# 进行可视化
# matplotlib.use('qt4agg')
#
# # 指定默认字体,解决matplot显示中文的问题
# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
# matplotlib.rcParams['font.family'] = ['KaiTi']
# matplotlib.pyplot.hist(count.values, 50, facecolor='g')
# matplotlib.pyplot.xlabel('出现次数')
# matplotlib.pyplot.ylabel('频数')
# matplotlib.pyplot.show()


'''统计各类别的信息'''
# ad_cols = ["Subject ID", "Group"]
# ad_data = ad_data[ad_cols]
#
# data_group = ad_data.groupby("Subject ID")
#
# for name, group in data_group:
#     print(name)
#     print(group)
#     # count = len(group["Group"].value_counts())
#     # print(group["Group"].value_counts())
#     # if len(group["Group"].value_counts()) > 1:
#     #     print("warning")
#     # print(len(group["Group"].value_counts()))


'''统计每一类别所占的数里'''
ad_cols = ["Subject ID", "Group"]
ad_data = ad_data[ad_cols]

data_group = ad_data.groupby("Subject ID").head(1)
CN = 0
MCI = 0
AD = 0

for index, row in data_group.iterrows():
    if row['Group'] == 'CN':
        CN += 1
    elif row['Group'] == 'MCI':
        MCI += 1
    else:
        AD += 1

print("The number of CN is {}. The number of MCI is {}. The number of AD is {}".format(str(CN), str(MCI), str(AD)))

