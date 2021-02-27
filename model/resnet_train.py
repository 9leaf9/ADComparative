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
from datasets.ADNI_spilt import AdniDataSet
from setting import parse_opts
from sklearn.metrics import roc_curve, auc
import time

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [5]))

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


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = self.fc1(x1.view(x1.shape[0], -1))
        x1 = self.fc2(x1)
        x = self.fc3(x1)

        return x


if __name__ == '__main__':
    sets = parse_opts()
    sets.gpu_id = [0]

    # data_path = '/data1/qiaohezhe/MRI_Score/new_ad_data_filter_mci.csv'
    # img_path = '/data1/qiaohezhe/MRI_Score/MRI/'

    # train_data = AdniDataSet(data_path, img_path, sets)
    """将训练集划分为训练集和验证集"""
    # train_db, val_db = torch.utils.data.random_split(train_data, [230, 101])
    # print('train:', len(train_db), 'validation:', len(val_db))

    # train_data_path1 = '/data1/qiaohezhe/MRI_Score/ad_nc/train_label_single.csv'
    # train_data_path2 = '/data1/qiaohezhe/MRI_Score/ad_nc/train_label_single_shuffle.csv'
    # val_data_path1 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized.csv'
    # val_data_path2 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized_shuffle.csv'

    # val_data_path1 = '/data1/qiaohezhe/AD2/adni2_ad_nc_val_label.csv'
    # val_data_path2 = '/data1/qiaohezhe/AD2/adni2_ad_nc_val_label_shuffle.csv'

    train_data_path1 = '/data1/qiaohezhe/MRI_Score/mci_nc/train_label_single2.csv'
    train_data_path2 = '/data1/qiaohezhe/MRI_Score/mci_nc/train_label_single_shuffle2.csv'
    val_data_path1 = '/data1/qiaohezhe/AD2/adni2_mci_nc_val_label.csv'
    val_data_path2 = '/data1/qiaohezhe/AD2/adni2_mci_nc_val_label_shuffle.csv'

    # train_img_path = '/data1/qiaohezhe/MRI_Score/ad_nc/train/'
    # val_img_path = '/data1/qiaohezhe/miriad/ad_nc_regsiter/'
    # val_img_path = '/data1/qiaohezhe/AD2/ad_nc_regsiter/'

    train_img_path = '/data1/qiaohezhe/MRI_Score/mci_nc/train/'
    val_img_path = '/data1/qiaohezhe/AD2/mci_nc_regsiter/'

    train_db = AdniDataSet(train_data_path1, train_img_path, sets)
    val_db = AdniDataSet(val_data_path1, val_img_path, sets)

    """将训练集划分为训练集和验证集"""

    train_loader = DataLoader(dataset=train_db, batch_size=12, shuffle=True)
    val_loader = DataLoader(dataset=val_db, batch_size=12, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes())

    pretrain = torch.load('/data1/qiaohezhe/ADNI/MRI_augment2/pre_train_model/resnet_50_23dataset.pth')
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if
                     k.replace('module.', '') in model.state_dict().keys()}
    # print(pretrain['state_dict'].keys())
    # print(model.state_dict().keys())
    print(pretrain_dict.keys())
    model.state_dict().update(pretrain_dict)
    model.load_state_dict(model.state_dict())

    model = torch.nn.DataParallel(model)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=60, verbose=True)
    model.to(device)
    result_list = []
    epochs = 100
    running_loss = 0
    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct = 0
        total = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
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
        roc_label = []
        roc_predict = []
        model.eval()
        with torch.no_grad():
            print("validation...")
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                _, predict = torch.max(output, 1)

                roc_label += labels.tolist()
                roc_output1 = torch.softmax(torch.transpose(output, 0, 1), dim=0)
                roc_output1 = roc_output1.tolist()
                roc_predict += roc_output1[1]
                print(labels.tolist())
                print(roc_output1[1])

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

            print(roc_label)
            print(roc_predict)
            fpr, tpr, threshold = roc_curve(roc_label, roc_predict, pos_label=1)  ###计算真正率和假正率
            roc_auc = auc(fpr, tpr)  ###计算auc的值

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
            result_list.append(
                [epoch, round(train_loss, 3), round(val_loss, 3), round(correct / total, 3), round(recall[1], 3), round(precision[1], 3), round(F1[1], 3),
                 round(roc_auc, 3)])

            # 输入日志
            name = ['epoch', 'train_loss', 'val_loss', 'val_acc', 'recall', 'precision', 'F1', 'AUC']
            result = pd.DataFrame(columns=name, data=result_list)

            # result.to_csv("/data1/qiaohezhe/MRI_Score/log/ad_nc_all/paper1/methods/resnet_baseline_mimrad.csv", mode='w',
            #               index=False,
            #               header=False)
            result.to_csv("/data1/qiaohezhe/MRI_Score/log/ad_nc_all/paper1/methods/resnet_baseline_adni2_mci.csv", mode='w',
                          index=False,
                          header=False)

            early_stopping(val_loss, model)


    stop = datetime.now()
    print("Running time: ", stop - start)
