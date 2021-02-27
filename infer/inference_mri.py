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

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [4, 5, 6]))

start = datetime.now()


# define my class to read data
class MyDataset(Dataset):
    def __init__(self, transform=None, img_path=None, label_path=None):
        self.transform = transform
        self.img_path = img_path
        self.label_path = label_path
        self.labels = pd.read_csv(self.label_path)
        self.images_file = h5py.File(self.img_path)
        self.images = self.images_file['data']

    def __getitem__(self, index):
        # 3d convolution
        # so don't reshape
        return self.images[index], self.labels['label'][index]

    def __len__(self):
        return len(self.images)


class FirstNet(nn.Module):

    def __init__(self, f=2):
        super(FirstNet, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                dilation=1))
        self.conv.add_module('conv2', nn.InstanceNorm3d(num_features=4 * f))
        self.conv.add_module('conv3', nn.ReLU(inplace=True))
        self.conv.add_module('conv4', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv5',
                             nn.Conv3d(in_channels=4 * f, out_channels=8 * f, kernel_size=3, stride=1, padding=0,
                                       dilation=2))
        self.conv.add_module('conv6', nn.InstanceNorm3d(num_features=8 * f))
        self.conv.add_module('conv7', nn.ReLU(inplace=True))
        self.conv.add_module('conv8', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv9',
                             nn.Conv3d(in_channels=8 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=2,
                                       dilation=2))
        self.conv.add_module('conv10', nn.InstanceNorm3d(num_features=16 * f))
        self.conv.add_module('conv11', nn.ReLU(inplace=True))
        self.conv.add_module('conv12', nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv13',
                             nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=2, stride=1, padding=1,
                                       dilation=2))
        self.conv.add_module('conv14', nn.InstanceNorm3d(num_features=32 * f))
        self.conv.add_module('conv15', nn.ReLU(inplace=True))
        self.conv.add_module('conv16', nn.MaxPool3d(kernel_size=5, stride=2))

        # self.conv.add_module('conv17',
        #                      nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
        #                                dilation=2))
        # self.conv.add_module('conv18', nn.InstanceNorm3d(num_features=64 * f))
        # self.conv.add_module('conv19', nn.ReLU(inplace=True))
        # self.conv.add_module('conv20', nn.MaxPool3d(kernel_size=5, stride=2))

        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(32 * f * 1 * 1 * 1, 64))
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.avgpool(x1)
        x2 = self.conv(x2)
        x2 = self.avgpool(x2)
        # print(x2.shape)
        x1 = self.fc(x1.view(x1.shape[0], -1))
        x1 = self.dropout(x1)
        x2 = self.fc(x2.view(x2.shape[0], -1))
        x2 = self.dropout(x2)
        x = torch.cat((x1, x2), 1)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = FirstNet(f=4)
    # model = torch.nn.DataParallel(model)

    # print(model)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.002)

    img_path = "/data1/qiaohezhe/ADNI/infer/002_S_0413/corr/001_processed.nii"
    mask_path = "/data1/qiaohezhe/ADNI/infer/002_S_0413/corr/" \
                "001_processed.nii_predicted_4class_unet3d_extract12_hist15.nii.gz"
    model_path = "/data1/qiaohezhe/ADNI/infer/model/c3d_drop6_mask_ad_mci_avg.pth"
    model = torch.load(model_path)
    print(model)
    model.to(device)
    train_data = AdniDataSet(img_path, mask_path)
    inputs, mask = train_data.get_mri_mask()
    inputs = torch.from_numpy(inputs).unsqueeze(dim=0)
    mask = torch.from_numpy(mask).unsqueeze(dim=0)
    print(inputs.shape)

    # with torch.no_grad():
    #     inputs, mask = inputs.to(device), mask.to(device)
    #     output = model(inputs, mask)
    #     _, predict = torch.max(output, 1)
    #     print("predict result", predict)

