# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/3/27
# version： Python 3.7.8
# @File : clincal_fcn.py
# @Software: PyCharm
'''临床数据  构建简单的全连接神经网络'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import gzip
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd


class ADdata(Dataset):
    def __init__(self, train_set, train_labels, transform=None):
        super(Dataset, self).__init__()
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        train_data, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            train_data = self.transform(train_data)
        return train_data, target

    def __len__(self):
        return len(self.train_set)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(10, 6)
        self.fc2 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y


if __name__ == "__main__":
    # 模型参数
    epoches = 100
    lr = 0.001

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
    train_data = ADdata(X, Y)

    """将训练集划分为训练集和验证集"""
    train_db, val_db = torch.utils.data.random_split(train_data, [688, 295])
    print('train:', len(train_db), 'validation:', len(val_db))

    train_loader = DataLoader(dataset=train_db, batch_size=8, shuffle=False)
    val_loader = DataLoader(dataset=val_db, batch_size=8, shuffle=False)
    print("Train data load success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet()
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    for k in model.parameters():
        print(type(k), k.shape)
    for k in model.named_parameters():
        print(isinstance(k, tuple), 'len{:d}'.format(len(k)), k[0], k[1].shape)

    loss = 0
    for epoch in range(epoches):
        for i, data in enumerate(train_loader):
            images, labels = data
            # images = images.reshape(-1, 28 * 28)
            images = torch.tensor(images, dtype=torch.float32)
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epoches, loss.item()))

        # 开始测试
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = torch.tensor(images, dtype=torch.float32)
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                _, predict = torch.max(output, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
            print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))

    # torch.save(model.state_dict(), "./model/neuralNet")



