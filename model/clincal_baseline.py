# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/18
# version： Python 3.7.8
# @File : clincal_baseline.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from joblib import dump
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy import interp
from joblib import load

seed = 8
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

data = pd.read_csv('../ADdata/AD_2020_7_31_7_30_2020_lastest_processed.csv')
col = ['scaled_Age', 'scaled_APOE A1', 'scaled_APOE A2', 'scaled_MMSE Total Score', 'scaled_GDSCALE Total Score', 'scaled_Global CDR',
       'scaled_Global CDR', 'scaled_NPI-Q Total Score', 'F', 'M', 'AD', 'CN', 'MCI']

# col = ['scaled_Age', 'scaled_APOE A1', 'scaled_APOE A2', 'scaled_MMSE Total Score', 'scaled_GDSCALE Total Score', 'scaled_Global CDR',
#        'scaled_Global CDR', 'scaled_NPI-Q Total Score', 'F', 'M', 'AD', 'CN', 'MCI']
data = data[col]
# data = data.fillna(data.mean())
# data.fillna(0, inplace=True)
data.dropna(axis=0, how='any', inplace=True)
'''二分类问题'''
data = data[~data['CN'].isin([1])]
# data = data[~data['AD'].isin([1])]
# data.drop(index=(data.loc[(data['CN'] == 1)].index))

# print("data", data['death'].value_counts())
Y = np.array(data['AD'])
X = data.drop(['AD', 'CN', 'MCI'], axis=1)
X = np.array(X)
print(X.shape)

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

k = 0
for train, validation in kfold.split(X, Y):

    # clf = RandomForestClassifier()
    # clf.fit(X[train], Y[train])

    # clf = XGBClassifier()
    # clf.fit(X[train], Y[train])

    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(X[train], Y[train])

    # y_score = clf.predict(X_test, 1)
    # Y_pred = clf.predict_proba(X)[:, 1]

    # clf = LogisticRegressionCV(cv=5, penalty='l2', tol=0.0001, fit_intercept=True, intercept_scaling=1,
    #                            class_weight=None, random_state=None,
    #                            max_iter=100, verbose=0, n_jobs=None).fit(X[train], Y[train])

    y_score = clf.predict_proba(X[validation])[:, 1]
    # 评估
    fpr, tpr, threshold = roc_curve(Y[validation], y_score, pos_label=1)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    aucs.append(roc_auc)
    tprs.append(interp(mean_fpr, fpr, tpr))  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    tprs[-1][0] = 0.0  # 初始处为0
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (k, roc_auc))
    print("roc_auc", roc_auc)
    k = k + 1


# # 画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)#计算平均AUC值
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
std_auc = np.std(tprs, axis=0)
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr-std_tpr, 0)
# plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# # shuffle and split train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1, random_state=1)
# print("start model...")
#
# clf = XGBClassifier()
# clf.fit(X_train, y_train)
# # clf = svm.SVC(kernel='rbf', probability=True)
# # clf.fit(X_train, y_train)
# # clf = RandomForestClassifier()
# # clf.fit(X_train, y_train)
# dump(clf, '../model/svm.joblib')
# print("model is done")