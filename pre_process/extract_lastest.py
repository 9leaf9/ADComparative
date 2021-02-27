# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/21
# version： Python 3.7.8
# @File : extract_lastest.py
# @Software: PyCharm

'''提取受试者最新得记录'''

import pandas as pd
ad_data = pd.read_csv('/data1/qiaohezhe/ADNI/ADdata/AD_2020_7_31_7_30_2020.csv')
ad_cols = ["Image Data ID", "Subject ID", "Group", "Sex", "Age", "Visit", "Acq Date"]
ad_data = ad_data[ad_cols]
clinical_data = pd.read_csv('/data1/qiaohezhe/ADNI/ADdata/idaSearch_7_30_2020.csv')
clinical_cols = ["Subject ID", "APOE A1", "APOE A2", "MMSE Total Score", "GDSCALE Total Score", "Global CDR", "FAQ Total Score", "NPI-Q Total Score"]
clinical_data = clinical_data[clinical_cols]
data = pd.merge(ad_data, clinical_data, on=['Subject ID'], how='left')
data.dropna(axis=0, how='any', inplace=True)
data = data.sort_values(['Subject ID', 'Age'], ascending=False).groupby('Subject ID').head(1)
print(data)

data.to_csv('/data1/qiaohezhe/ADNI/ADdata/AD_2020_7_31_7_30_2020_latest.csv', index=False)
