# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/10/12
# version： Python 3.7.8
# @File : inference_mri.py
# @Software: PyCharm
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import cat
import os
import torch.nn.init as init
import math
import sys
import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime
from ADNI_infer import AdniDataSet
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [4, 5, 6]))

start = datetime.now()


class FirstNet(nn.Module):

    def __init__(self, f=8):
        super(FirstNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                  dilation=1))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=0,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm3d(num_features=16 * f))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer4.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer4.add_module('relu4', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))

        self.layer5 = nn.Sequential()
        self.layer5.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer5.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer5.add_module('relu3', nn.ReLU(inplace=True))
        self.layer5.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer6 = nn.Sequential()
        self.layer6.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer6.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer6.add_module('relu4', nn.ReLU(inplace=True))
        self.layer6.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))

        # self.layer7 = nn.Sequential()
        # self.layer7.add_module('conv3',
        #                        nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
        #                                  dilation=2))
        # self.layer7.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        # self.layer7.add_module('relu3', nn.ReLU(inplace=True))
        # self.layer7.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))
        #
        # self.layer8 = nn.Sequential()
        # self.layer8.add_module('conv4',
        #                        nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
        #                                  dilation=2))
        # self.layer8.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        # self.layer8.add_module('relu4', nn.ReLU(inplace=True))
        # self.layer8.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(64 * f, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        self.fc4 = nn.Linear(512 * 3 * 3 * 3, 256)
        self.fc5 = nn.Linear(256, 3)

        self.fc7 = nn.Linear(512 * 3 * 3 * 3, 256)
        self.fc8 = nn.Linear(256, 64)
        self.fc9 = nn.Linear(64, 1)

    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x2 = self.layer1(x2)

        x1 = self.layer2(x1)
        x2 = self.layer2(x2)

        x1 = self.layer3(x1)
        x2 = self.layer3(x2)
        x_s1 = x1
        x_s2 = x2
        x1_embedding = x1
        # x1_embedding = F.normalize(x1_embedding)

        x2_embedding = x2
        # x2_embedding = F.normalize(x2_embedding)
        x1 = self.layer4(x1)
        x2 = self.layer4(x2)
        x_score1 = x1
        x_score2 = x2

        x_score = x_score1 - x_score2
        x_score = x_score.view(x_score.shape[0], -1)

        x_score = self.fc4(x_score)
        x_s = self.fc5(x_score)

        x_s1 = self.layer6(x_s1)
        x_s2 = self.layer6(x_s2)
        x_s1 = x_s1.view(x_s1.shape[0], -1)
        x_s2 = x_s2.view(x_s2.shape[0], -1)

        x_s1 = self.fc7(x_s1)
        x_s1 = self.fc8(x_s1)
        x_s1 = self.fc9(x_s1)

        x_s2 = self.fc7(x_s2)
        x_s2 = self.fc8(x_s2)
        x_s2 = self.fc9(x_s2)

        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)

        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)

        x2 = self.fc1(x2)
        x2 = self.fc2(x2)
        x2 = self.fc3(x2)

        x_s1 = x_s1.reshape(x_s1.shape[0])
        x_s2 = x_s2.reshape(x_s2.shape[0])

        x_embedding1 = x1_embedding.view(x1_embedding.size()[0], -1)
        x_embedding2 = x2_embedding.view(x2_embedding.size()[0], -1)

        return x1, x2, x_embedding1, x_embedding2, x_s, x_s1, x_s2


def trasfer_label(group_list):
    label_list = []
    for group in group_list:
        if group == 'AD':
            label = 1
        else:
            label = 0
        label_list.append(label)
    return label_list


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_path1 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized.csv'
    train_data_path2 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized_shuffle.csv'

    train_img_path = '/data1/qiaohezhe/miriad/ad_nc_regsiter/'

    model_path = "/data1/qiaohezhe/MRI_Score/log/ad_nc_all/model/group/c3d_score100/c3d_score54.pth"

    model = torch.load(model_path)
    # print(model)
    # parm = {}
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    #     parm[name] = parameters.detach().cpu().numpy()
    # print(parm['module.fc1.fc1.weight'])
    # print(parm['module.fc2.fc2.weight'])
    # print(parm['module.fc3.fc3.weight'])

    model.to(device)
    data1 = pd.read_csv(train_data_path1)
    data2 = pd.read_csv(train_data_path2)
    mri_list1 = data1['index']
    mri_list2 = data2['index']

    label_list1 = trasfer_label(data1['group'].tolist())
    label_list2 = trasfer_label(data2['group'].tolist())
    result_list = []

    for mri1, mri2, label1, label2 in zip(mri_list1, mri_list2, label_list1, label_list2):
        print("predicting {} ".format(mri1))
        img_path1 = train_img_path + str(mri1) + ".nii"
        img_path2 = train_img_path + str(mri2) + ".nii"
        train_data = AdniDataSet(img_path1, img_path2)
        inputs1, inputs2 = train_data.get_mri()

        inputs1 = torch.from_numpy(inputs1).unsqueeze(dim=0)
        inputs2 = torch.from_numpy(inputs2).unsqueeze(dim=0)
        model.eval()
        with torch.no_grad():
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
            logps1, logps2, embedding1, embedding2, logps_s, logps_score1, logps_score2 = model(inputs1, inputs2)
            output1 = torch.softmax(torch.transpose(logps1, 0, 1), dim=0)
            # output2 = torch.softmax(torch.transpose(logps2, 0, 1), dim=0)
            # output = torch.sigmoid(output)
            # print("predict result", output[0, 1].item())
            result_list.append([mri1, label1, output1[1, 0].item()])

    name = ['subject', 'group', 'predict_group']
    data_csv = pd.DataFrame(columns=name, data=result_list)
    data_csv.to_csv("/data1/qiaohezhe/MRI_Score/log/ad_nc_all/infer/group/c3d_score100.csv", index=False)
