# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/24
# version： Python 3.7.8
# @File : multi_svm.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score

seed = 8
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
data = pd.read_csv('../ADdata/AD_2020_7_31_7_30_2020_lastest_processed.csv')
col = ['scaled_Age', 'scaled_APOE A1', 'scaled_APOE A2', 'scaled_MMSE Total Score', 'scaled_GDSCALE Total Score',
       'scaled_Global CDR',
       'scaled_Global CDR', 'scaled_NPI-Q Total Score', 'F', 'M', 'AD', 'CN', 'MCI']
data = data[col]
# data = data.fillna(data.mean())
# data.fillna(0, inplace=True)
data.dropna(axis=0, how='any', inplace=True)
'''三分类  label标记'''
true_index = []
for index, value in enumerate(data['AD'].isin([1])):
    if value is True:
        true_index.append(index)
'''AD 2    MCI 1   NC 0'''
data.loc[true_index, 'CN'] = 2
# print(data)
Y = np.array(data['CN'])
X = data.drop(['AD', 'CN', 'MCI'], axis=1)
X = np.array(X)




k = 0
for train, validation in kfold.split(X, Y):

    # clf = RandomForestClassifier()
    # clf.fit(X[train], Y[train])

    # clf = XGBClassifier()
    # clf.fit(X[train], Y[train])

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X, Y)

    # y_score = clf.predict(X_test, 1)
    # Y_pred = clf.predict_proba(X)[:, 1]

    # clf = LogisticRegressionCV(cv=5, penalty='l2', tol=0.0001, fit_intercept=True, intercept_scaling=1,
    #                            class_weight=None, random_state=None,
    #                            max_iter=100, verbose=0, n_jobs=None).fit(X[train], Y[train])


    predict_y = clf.predict(X[validation])
    ac = float((Y[validation] == predict_y).sum()) / (len(Y[validation]))
    print("accuracy of SVM:", ac)
    print(f1_score(Y[validation], predict_y, average='weighted'))



    # y_score = clf.predict_proba(X[validation])[:, 1]
    # # 评估
    # fpr, tpr, threshold = roc_curve(Y[validation], y_score, pos_label=1)  ###计算真正率和假正率
    # roc_auc = auc(fpr, tpr)  ###计算auc的值
    # aucs.append(roc_auc)
    # tprs.append(interp(mean_fpr, fpr, tpr))  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    # tprs[-1][0] = 0.0  # 初始处为0
    # #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (k, roc_auc))
    # print("roc_auc", roc_auc)