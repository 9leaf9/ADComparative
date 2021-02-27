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
from datasets.ADNI_compare_score3 import AdniDataSet_train
from datasets.ADNI_compare_score3 import AdniDataSet_val
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

        self.fc4 = nn.Linear(512 * 3 * 3 * 3, 256)
        self.fc5 = nn.Linear(256, 3)

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

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        if not self.no_max_pool:
            x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x_score1 = x1
        x_score2 = x2
        x_score = x_score1 - x_score2
        x_score = x_score.view(x_score.shape[0], -1)

        x_score = self.fc4(x_score)
        x_s = self.fc5(x_score)

        x1 = self.avgpool(x1)
        x1 = self.fc1(x1.view(x1.shape[0], -1))
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)

        x2 = self.avgpool(x2)
        x2 = self.fc1(x2.view(x2.shape[0], -1))
        x2 = self.fc2(x2)
        x2 = self.fc3(x2)

        return x1, x2, x_s


if __name__ == '__main__':
    sets = parse_opts()
    sets.gpu_id = [0]



    train_data_path1 = '/data1/qiaohezhe/MRI_Score/ad_nc/train_label_single.csv'
    train_data_path2 = '/data1/qiaohezhe/MRI_Score/ad_nc/train_label_single_shuffle.csv'
    # val_data_path1 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized.csv'
    # val_data_path2 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized_shuffle.csv'

    val_data_path1 = '/data1/qiaohezhe/AD2/adni2_ad_nc_val_label.csv'
    val_data_path2 = '/data1/qiaohezhe/AD2/adni2_ad_nc_val_label_shuffle.csv'

    train_img_path = '/data1/qiaohezhe/MRI_Score/ad_nc/train/'
    # val_img_path = '/data1/qiaohezhe/miriad/ad_nc_regsiter/'
    val_img_path = '/data1/qiaohezhe/AD2/ad_nc_regsiter/'

    train_db = AdniDataSet_train(train_data_path1, train_data_path2, train_img_path, sets)
    val_db = AdniDataSet_val(val_data_path1, val_data_path2, val_img_path, sets)
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
    print(model)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=30, verbose=True)
    model.to(device)
    result_list = []
    epochs = 110
    running_group_loss = 0
    running_loss = 0
    running_compare_loss = 0
    running_score_loss = 0
    running_contrastive_loss = 0
    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs1, inputs2, labels1, labels2, labels3, labels4, score1, score2 = data
            inputs1, inputs2, labels1, labels2, labels3, labels4 = inputs1.to(device), inputs2.to(
                device), labels1.to(device), labels2.to(device), labels3.to(device), labels4.to(device)
            score1, score2 = score1.to(device), score2.to(device)
            optimizer.zero_grad()

            logps1, logps2, logps_s = model.forward(inputs1, inputs2)

            print(logps_s)

            loss1 = criterion(logps1, labels1)
            loss2 = criterion(logps2, labels2)
            loss_group = loss1 + loss2
            print("loss_group", loss_group)

            loss_compare = criterion(logps_s, labels4)
            print("loss_compare", loss_compare)
            print(logps_score1.shape)

            loss = 0.8 * loss_group + 0.2 * loss_compare
            print("loss", loss)

            _, predict1 = torch.max(logps1, 1)
            _, predict2 = torch.max(logps2, 1)


            print("train group ground truth", labels1)
            print("train group predict", predict1)

            print("train score ground truth1", score1)
            print("train score predict1", logps_score1)

            print("train score ground truth2", score2)
            print("train score predict2", logps_score2)

            print("train compare ground truth", labels4)
            print("train compare predict", logps_s)

            correct1 += (predict1 == labels1).sum().item()
            total1 += labels1.size(0)
            correct2 += (predict2 == labels2).sum().item()
            total2 += labels2.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_group_loss += loss_group.item()
            running_score_loss += loss_score.item()
            running_compare_loss += loss_compare.item()

        train_loss = running_loss / len(train_loader)
        train_group_loss = running_group_loss / len(train_loader)
        train_score_loss = running_score_loss / len(train_loader)
        train_compare_loss = running_compare_loss / len(train_loader)

        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, running_loss / len(train_loader)))
        print("The accuracy of total {} images: {}%".format(total1, 100 * correct1 / total1))
        print("The accuracy of total {} images: {}%".format(total2, 100 * correct2 / total2))

        running_group_loss = 0
        running_loss = 0
        running_compare_loss = 0
        running_score_loss = 0
        running_contrastive_loss = 0
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        classnum = 2
        target_num = torch.zeros((1, classnum))
        predict_num = torch.zeros((1, classnum))
        acc_num = torch.zeros((1, classnum))
        roc_label = []
        roc_predict = []
        model.eval()
        with torch.no_grad():
            print("validation...")
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                inputs1, inputs2, labels1, labels2, labels3, labels4, score1, score2 = data
                inputs1, inputs2, labels1, labels2, labels3, labels4, score1, score2 = inputs1.to(device), inputs2.to(
                    device), \
                                                                                       labels1.to(device), labels2.to(
                    device), labels3.to(device), labels4.to(device), score1.to(device), score2.to(device)
                output1, output2,  logps_s = model.forward(inputs1, inputs2)

                loss1 = criterion(output1, labels1)
                loss2 = criterion(output2, labels2)
                loss_group = loss1 + loss2
                print("loss_group", loss_group)
                loss_compare = criterion(logps_s, labels4)

                print("loss_compare", loss_compare)

                loss_score = 0.5 * loss3 + 0.5 * loss4
                print("loss_score", loss_score)

                loss = 0.8 * loss_group + 0.2 * loss_compare
                print("loss", loss)

                _, predict1 = torch.max(output1, 1)
                _, predict2 = torch.max(output2, 1)

                roc_label += labels1.tolist()
                roc_output1 = torch.softmax(torch.transpose(output1, 0, 1), dim=0)
                roc_output1 = roc_output1.tolist()
                roc_predict += roc_output1[1]
                print(labels1.tolist())
                print(roc_output1[1])

                print("train group ground truth", labels1)
                print("train group predict", predict1)

                print("train score ground truth", score1)
                print("train score predict", logps_score1)

                print("train compare ground truth", labels4)
                print("train compare predict", logps_s)


                running_loss += loss.item()
                running_group_loss += loss_group.item()
                running_score_loss += loss_score.item()
                running_compare_loss += loss_compare.item()

                total1 += labels1.size(0)
                total2 += labels2.size(0)

                correct1 += (predict1 == labels1).sum().item()
                correct2 += (predict2 == labels2).sum().item()

                '''calculate  Recall Precision F1'''
                pre_mask = torch.zeros(output1.size()).scatter_(1, predict1.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                tar_mask = torch.zeros(output1.size()).scatter_(1, labels1.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)

            val_loss = running_loss / len(val_loader)
            val_group_loss = running_group_loss / len(val_loader)
            val_score_loss = running_score_loss / len(val_loader)
            val_compare_loss = running_compare_loss / len(val_loader)

            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            # 精度调整
            recall = (recall.numpy()[0] * 100).round(3)
            precision = (precision.numpy()[0] * 100).round(3)
            F1 = (F1.numpy()[0] * 100).round(3)

            print(roc_label)
            print(roc_predict)
            fpr, tpr, threshold = roc_curve(roc_label, roc_predict, pos_label=1)  ###计算真正率和假正率
            roc_auc = auc(fpr, tpr)  ###计算auc的值

            print("The accuracy of valid {} images: {}%".format(total1, 100 * correct1 / total1))
            print(
                "The accuracy of valid {} images: recall {}, precision {}, F1 {}".format(total1, recall, precision, F1))
            result_list.append(
                [epoch, round(train_loss, 3), round(val_loss, 3), round(correct1 / total1, 3),
                 round(correct2 / total2, 3), round(recall[1], 3), round(precision[1], 3), round(F1[1], 3),
                 round(roc_auc, 3),
                 round(train_group_loss, 3), round(train_compare_loss, 3),
                 round(val_group_loss, 3),
                 round(val_compare_loss, 3)])

            # 输入日志
            name = ['epoch', 'train_loss', 'val_loss', 'val_acc1', 'val_acc2', 'recall', 'precision', 'F1', 'AUC',
                    'train_group_loss', 'train_compare_loss', 'val_group_loss',
                    'val_compare_loss']
            result = pd.DataFrame(columns=name, data=result_list)

            result.to_csv("/data1/qiaohezhe/MRI_Score/log/ad_nc_all/paper1/methods/resnet_score820_score0.csv", mode='w', index=False,
                          header=False)
            early_stopping(val_loss, model)
