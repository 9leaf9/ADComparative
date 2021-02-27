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

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [6,7,8]))
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

        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(64 * f, 256))
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.sw1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(4, 8, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        )

        self.sw2 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(1, 1, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(1, 1, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        )
        self.sw3 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(1, 1, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(1, 1, kernel_size=(1, 3, 3)),
            nn.BatchNorm3d(num_features=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        )

    def forward(self, x1, x2):
        x1_sw1 = self.sw1(x1)
        # x1_sw1 = torch.squeeze(x1_sw1)
        x1_sw1 = x1_sw1.transpose(1, 2)
        x1_sw1 = x1_sw1.contiguous().view(x1_sw1.shape[0], x1_sw1.shape[1], -1)
        x1_corr1 = torch.bmm(x1_sw1, x1_sw1.transpose(1, 2))
        x1_corr1 = torch.mean(x1_corr1, 2)
        x1_corr1 = torch.softmax(x1_corr1, 1).view(x1_corr1.shape[0], x1_corr1.shape[1], 1)
        att_x1 = x1
        att_x1 = torch.squeeze(att_x1)
        att_x1 = att_x1.view(att_x1.shape[0], att_x1.shape[1], -1)
        # print(att_x1.shape)
        # print(x1_corr1.shape)
        att_x1 = x1_corr1 * att_x1
        att_x1 = att_x1.view(att_x1.shape[0], 1, att_x1.shape[1], 90, 90)
        # x1_corr1 = x1_corr1.view(x1_corr1.shape[0], 1, x1_corr1.shape[1], 1, 1)

        # x1_sw2 = self.sw2(x1.transpose(2, 3))
        # x1_sw2 = torch.squeeze(x1_sw2)
        # x1_sw2 = x1_sw2.view(x1_sw2.shape[0], x1_sw2.shape[1], -1)
        # x1_corr2 = torch.bmm(x1_sw2, x1_sw2.transpose(1, 2))
        # x1_corr2 = torch.mean(x1_corr2, 2)
        # x1_corr2 = torch.softmax(x1_corr2, 1)
        # x1_corr2 = x1_corr2.view(x1_corr2.shape[0], 1, x1_corr2.shape[1], 1, 1)
        #
        # x1_sw3 = self.sw3(x1.transpose(2, 4))

        # x1_sw3 = torch.squeeze(x1_sw3)
        # x1_sw3 = x1_sw1.view(x1_sw3.shape[0], x1_sw3.shape[1], -1)
        # x1_corr3 = torch.bmm(x1_sw3, x1_sw3.transpose(1, 2))
        # x1_corr3 = torch.mean(x1_corr3, 2)
        # x1_corr3 = torch.softmax(x1_corr3, 1)
        # x1_corr3 = x1_corr3.view(x1_corr3.shape[0], 1, x1_corr3.shape[1], 1, 1)
        # print(att_x1.shape)

        x1 = att_x1
        # x1 = x1 + x1 * x1_corr2.transpose(2, 3)
        # x1 = x1 + x1 * x1_corr3.transpose(2, 4)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = self.fc(x1.view(x1.shape[0], -1))
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        return x1


if __name__ == "__main__":
    sets = parse_opts()
    sets.gpu_id = [0]

    # train_data_path = '/data1/qiaohezhe/ADNI/MRI_augment2/ad_vs_mci/train_label.csv'
    # val_data_path = '/data1/qiaohezhe/ADNI/MRI_augment2/ad_vs_mci/val_label.csv'
    #
    # train_img_path = '/data1/qiaohezhe/ADNI/MRI_augment2/ad_vs_mci/train/'
    # val_img_path = '/data1/qiaohezhe/ADNI/MRI_augment2/ad_vs_mci/val/'

    train_data_path = '/data1/qiaohezhe/MRI_Score/mci_nc/train_label.csv'
    val_data_path = '/data1/qiaohezhe/MRI_Score/mci_nc/val_label.csv'

    train_img_path = '/data1/qiaohezhe/MRI_Score/mci_nc/train/'
    val_img_path = '/data1/qiaohezhe/MRI_Score/mci_nc/val/'

    train_data = AdniDataSet(train_data_path, train_img_path, sets)
    val_data = AdniDataSet(val_data_path, val_img_path, sets)
    """将训练集划分为训练集和验证集"""

    # train_db, val_db = torch.utils.data.random_split(train_data, [700, 180])
    print('train:', len(train_data), 'validation:', len(val_data))

    train_loader = DataLoader(dataset=train_data, batch_size=24, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=24, shuffle=True)
    print("Train data load success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    model.to(device)
    result_list = []
    epochs = 80
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
            # result.to_csv("/data1/qiaohezhe/ADNI/log/ad_vs_mci/c3d_mri_epoch200_bn24_baseline_rate.csv", mode='w', index=False,
            #               header=False)
            # torch.save(model, "/data1/qiaohezhe/ADNI/log/ad_vs_mci/model/c3d/c3d_mri_epoch200_bn24_{}.pth".format(epoch))

            result.to_csv("/data1/qiaohezhe/MRI_Score/log/mci_nc/mri_attention3.csv", mode='w', index=False,
                          header=False)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/mri_attention3/mri_attention.pth")
                print("Early stopping")
                break
            torch.save(model,
                       "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/mri_attention3/mri_attention{}.pth".format(epoch))
    torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/mri_attention3/mri_attention.pth")

    stop = datetime.now()
    print("Running time: ", stop - start)
