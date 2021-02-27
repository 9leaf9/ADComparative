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
from datasets.ADNI import AdniDataSet
from setting import parse_opts
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [8]))
start = datetime.now()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class FirstNet(nn.Module):

    def __init__(self, f=8):
        super(FirstNet, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1',
                               nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=(90, 3, 3), stride=1, padding=0,
                                         dilation=1))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv1',
                               nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=(3, 90, 3), stride=1, padding=0,
                                         dilation=1))
        self.layer2.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer2.add_module('relu1', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling1', nn.MaxPool3d(kernel_size=(3, 1, 3), stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv1',
                               nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=(3, 3, 90), stride=1, padding=0,
                                         dilation=1))
        self.layer3.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer3.add_module('relu1', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling1', nn.MaxPool3d(kernel_size=(3, 3, 1), stride=2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv1', nn.Conv2d(in_channels=4 * f, out_channels=16 * f, kernel_size=(3, 3), stride=1,
                                                  padding=0, dilation=1))
        self.layer4.add_module('bn1', nn.BatchNorm2d(num_features=16 * f))
        self.layer4.add_module('relu1', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer5 = nn.Sequential()
        self.layer5.add_module('conv1', nn.Conv2d(in_channels=16 * f, out_channels=32 * f, kernel_size=(3, 3), stride=1,
                                                  padding=0, dilation=1))
        self.layer5.add_module('bn1', nn.BatchNorm2d(num_features=32 * f))
        self.layer5.add_module('relu1', nn.ReLU(inplace=True))
        self.layer5.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer6 = nn.Sequential()
        self.layer6.add_module('conv1', nn.Conv2d(in_channels=32 * f, out_channels=64 * f, kernel_size=(3, 3), stride=1,
                                                  padding=0, dilation=1))
        self.layer6.add_module('bn1', nn.BatchNorm2d(num_features=64 * f))
        self.layer6.add_module('relu1', nn.ReLU(inplace=True))
        self.layer6.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer7 = nn.Sequential()
        self.layer7.add_module('conv1', nn.Conv2d(in_channels=4 * f, out_channels=16 * f, kernel_size=(3, 3), stride=1,
                                                  padding=0, dilation=1))
        self.layer7.add_module('bn1', nn.BatchNorm2d(num_features=16 * f))
        self.layer7.add_module('relu1', nn.ReLU(inplace=True))
        self.layer7.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer8 = nn.Sequential()
        self.layer8.add_module('conv1', nn.Conv2d(in_channels=16 * f, out_channels=32 * f, kernel_size=(3, 3), stride=1,
                                                  padding=0, dilation=1))
        self.layer8.add_module('bn1', nn.BatchNorm2d(num_features=32 * f))
        self.layer8.add_module('relu1', nn.ReLU(inplace=True))
        self.layer8.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer9 = nn.Sequential()
        self.layer9.add_module('conv1', nn.Conv2d(in_channels=32 * f, out_channels=64 * f, kernel_size=(3, 3), stride=1,
                                                  padding=0, dilation=1))
        self.layer9.add_module('bn1', nn.BatchNorm2d(num_features=64 * f))
        self.layer9.add_module('relu1', nn.ReLU(inplace=True))
        self.layer9.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer10 = nn.Sequential()
        self.layer10.add_module('conv1', nn.Conv2d(in_channels=4 * f, out_channels=16 * f, kernel_size=(3, 3), stride=1,
                                                   padding=0, dilation=1))
        self.layer10.add_module('bn1', nn.BatchNorm2d(num_features=16 * f))
        self.layer10.add_module('relu1', nn.ReLU(inplace=True))
        self.layer10.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer11 = nn.Sequential()
        self.layer11.add_module('conv1',
                                nn.Conv2d(in_channels=16 * f, out_channels=32 * f, kernel_size=(3, 3), stride=1,
                                          padding=0, dilation=1))
        self.layer11.add_module('bn1', nn.BatchNorm2d(num_features=32 * f))
        self.layer11.add_module('relu1', nn.ReLU(inplace=True))
        self.layer11.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.layer12 = nn.Sequential()
        self.layer12.add_module('conv1',
                                nn.Conv2d(in_channels=32 * f, out_channels=64 * f, kernel_size=(3, 3), stride=1,
                                          padding=0, dilation=1))
        self.layer12.add_module('bn1', nn.BatchNorm2d(num_features=64 * f))
        self.layer12.add_module('relu1', nn.ReLU(inplace=True))
        self.layer12.add_module('max_pooling1', nn.MaxPool2d(kernel_size=(3, 3), stride=2))

        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.se1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16 * f, 16 * f // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16 * f // 16, 16 * f, kernel_size=1),
            nn.Sigmoid()
        )

        self.se2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16 * f, 16 * f // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16 * f // 16, 16 * f, kernel_size=1),
            nn.Sigmoid()
        )

        self.se3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(16 * f, 16 * f // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16 * f // 16, 16 * f, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1_conv1 = self.layer1(x1)
        x1_conv2 = self.layer2(x1)
        x1_conv3 = self.layer3(x1)

        x1_conv1 = torch.squeeze(x1_conv1)
        x1_conv2 = torch.squeeze(x1_conv2)
        x1_conv3 = torch.squeeze(x1_conv3)

        x1_conv1 = self.layer4(x1_conv1)
        x1_conv1_se = self.se1(x1_conv1)
        x1_conv1 = x1_conv1 * x1_conv1_se
        x1_conv1 = self.layer5(x1_conv1)
        x1_conv1 = self.layer6(x1_conv1)

        x1_conv2 = self.layer4(x1_conv2)
        x1_conv2_se = self.se2(x1_conv2)
        x1_conv2 = x1_conv2 * x1_conv2_se
        x1_conv2 = self.layer5(x1_conv2)
        x1_conv2 = self.layer6(x1_conv2)

        x1_conv3 = self.layer4(x1_conv3)
        x1_conv3_se = self.se3(x1_conv3)
        x1_conv3 = x1_conv3 * x1_conv3_se
        x1_conv3 = self.layer5(x1_conv3)
        x1_conv3 = self.layer6(x1_conv3)

        x1_conv1 = self.avgpool(x1_conv1)
        x1_conv2 = self.avgpool(x1_conv2)
        x1_conv3 = self.avgpool(x1_conv3)

        x1_conv1 = x1_conv1.view(x1_conv1.shape[0], -1)
        x1_conv2 = x1_conv2.view(x1_conv2.shape[0], -1)
        x1_conv3 = x1_conv3.view(x1_conv3.shape[0], -1)

        x = torch.stack((x1_conv1, x1_conv2, x1_conv3), 1)
        x = torch.mean(x, 1)
        x = torch.squeeze(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x


if __name__ == "__main__":
    sets = parse_opts()
    sets.gpu_id = [0]

    train_data_path = '/data1/qiaohezhe/MRI_Score/mci_nc/train_label.csv'
    val_data_path = '/data1/qiaohezhe/MRI_Score/mci_nc/val_label.csv'

    # train_data_path = '/data1/qiaohezhe/MRI_Score/ad_nc/train_label.csv'
    # val_data_path = '/data1/qiaohezhe/MRI_Score/ad_nc/val_label.csv'
    # #
    train_img_path = '/data1/qiaohezhe/MRI_Score/mci_nc/train/'
    val_img_path = '/data1/qiaohezhe/MRI_Score/mci_nc/val/'

    # train_img_path = '/data1/qiaohezhe/MRI_Score/ad_nc/train/'
    # val_img_path = '/data1/qiaohezhe/MRI_Score/ad_nc/val/'

    train_data = AdniDataSet(train_data_path, train_img_path, sets)
    val_data = AdniDataSet(val_data_path, val_img_path, sets)
    """将训练集划分为训练集和验证集"""

    # train_db, val_db = torch.utils.data.random_split(train_data, [700, 180])
    print('train:', len(train_data), 'validation:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=8, shuffle=True)
    print("Train data load success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    early_stopping = EarlyStopping(patience=60, verbose=True)
    model.to(device)
    result_list = []
    epochs = 60
    running_loss = 0
    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct = 0
        total = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, mask, labels = data
            inputs, mask, labels = inputs.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs, mask)
            loss = criterion(logps, labels)
            _, predict = torch.max(logps, 1)
            print("train ground truth", labels)
            print("train predict", predict)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, running_loss / len(train_loader)))
        print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))
        running_loss = 0
        correct = 0
        total = 0
        classnum = 2
        target_num = torch.zeros((1, classnum))
        predict_num = torch.zeros((1, classnum))
        acc_num = torch.zeros((1, classnum))
        model.eval()
        with torch.no_grad():
            print("validation...")
            for inputs, mask, labels in val_loader:
                inputs, mask, labels = inputs.to(device), mask.to(device), labels.to(device)
                output = model(inputs, mask)
                _, predict = torch.max(output, 1)
                loss = criterion(output, labels)
                running_loss += loss.item()
                total += labels.size(0)
                print("valid ground truth", labels)
                print("valid predict", predict)
                correct += (predict == labels).sum().item()
                '''calculate  Recall Precision F1'''
                pre_mask = torch.zeros(output.size()).scatter_(1, predict.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                tar_mask = torch.zeros(output.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)

            val_loss = running_loss / len(val_loader)
            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            # 精度调整
            recall = (recall.numpy()[0] * 100).round(3)
            precision = (precision.numpy()[0] * 100).round(3)
            F1 = (F1.numpy()[0] * 100).round(3)

            print("The accuracy of valid {} images: {}%".format(total, 100 * correct / total))
            print(
                "The accuracy of valid {} images: recall {}, precision {}, F1 {}".format(total, recall, precision, F1))
            result_list.append([epoch, train_loss, val_loss, correct / total, recall[1], precision[1], F1[1]])

            # 输入日志
            name = ['epoch', 'train_loss', 'val_loss', 'val_acc', 'recall', 'precision', 'F1']
            result = pd.DataFrame(columns=name, data=result_list)

            result.to_csv("/data1/qiaohezhe/MRI_Score/log/mci_nc/c3d_conv.csv", mode='w', index=False,
            header=False)
            # result.to_csv("/data1/qiaohezhe/MRI_Score/log/ad_nc/c3d_conv.csv", mode='w', index=False,
            #               header=False)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_conv/c3d_conv_final.pth")
                print("Early stopping")
                break
            # torch.save(model,
            # "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_conv/c3d_conv_{}.pth".format(epoch))
    # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_conv/c3d_conv_final.pth")

    stop = datetime.now()
    print("Running time: ", stop - start)
