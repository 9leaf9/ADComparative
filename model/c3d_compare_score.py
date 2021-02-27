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
from datasets.ADNI_compare_score import AdniDataSet
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


# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """
#
#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#
#         return loss_contrastive

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


        self.layer5 = nn.Sequential()
        self.layer5.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                  dilation=1))
        self.layer5.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer5.add_module('relu1', nn.ReLU(inplace=True))
        self.layer5.add_module('max_pooling1', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer6 = nn.Sequential()
        self.layer6.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=0,
                                         dilation=2))
        self.layer6.add_module('bn2', nn.BatchNorm3d(num_features=16 * f))
        self.layer6.add_module('relu2', nn.ReLU(inplace=True))
        self.layer6.add_module('max_pooling2', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer7 = nn.Sequential()
        self.layer7.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer7.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer7.add_module('relu3', nn.ReLU(inplace=True))
        self.layer7.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer8 = nn.Sequential()
        self.layer8.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer8.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer8.add_module('relu4', nn.ReLU(inplace=True))
        self.layer8.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(64 * f, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

        self.fc4 = nn.Linear(64 * f, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1_embedding = x1

        x1_embedding = F.normalize(x1_embedding)
        x1 = self.layer4(x1)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2_embedding = x2

        x2_embedding = F.normalize(x2_embedding)
        x2 = self.layer4(x2)

        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)

        x1_embedding = x1_embedding.view(x1_embedding.size()[0], -1)
        x2_embedding = x2_embedding.view(x2_embedding.size()[0], -1)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)

        x1_score = x1
        x2_score = x2

        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)

        x2 = self.fc1(x2)
        x2 = self.fc2(x2)
        x2 = self.fc3(x2)

        x1_score = self.fc4(x1_score)
        x1_score = self.fc5(x1_score)
        x1_score = self.fc6(x1_score)

        x2_score = self.fc4(x2_score)
        x2_score = self.fc5(x2_score)
        x2_score = self.fc6(x2_score)
        x_s = F.sigmoid(x1_score - x2_score)
        x_s = torch.stack((x_s, 1-x_s), 1)
        x_s = x_s.view(x_s.shape[0], 2)
        # print(x_s.shape)

        return x1, x2, x1_embedding, x2_embedding, x_s


if __name__ == "__main__":
    sets = parse_opts()
    sets.gpu_id = [0]

    # data_path = '/data1/qiaohezhe/MRI_Score/new_ad_data_filter_mci.csv'
    # img_path = '/data1/qiaohezhe/MRI_Score/MRI/'

    # train_data = AdniDataSet(data_path, img_path, sets)
    """将训练集划分为训练集和验证集"""
    # train_db, val_db = torch.utils.data.random_split(train_data, [230, 101])
    # print('train:', len(train_db), 'validation:', len(val_db))

    train_data_path1 = '/data1/qiaohezhe/MRI_Score/ad_nc/train_label_single.csv'
    train_data_path2 = '/data1/qiaohezhe/MRI_Score/ad_nc/train_label_single_shuffle.csv'
    val_data_path1 = '/data1/qiaohezhe/MRI_Score/ad_nc/val_label.csv'
    val_data_path2 = '/data1/qiaohezhe/MRI_Score/ad_nc/val_label_shuffle.csv'

    train_img_path = '/data1/qiaohezhe/MRI_Score/ad_nc/train/'
    val_img_path = '/data1/qiaohezhe/MRI_Score/ad_nc/val/'

    train_db = AdniDataSet(train_data_path1, train_data_path2, train_img_path, sets)
    val_db = AdniDataSet(val_data_path1, val_data_path2, val_img_path, sets)
    """将训练集划分为训练集和验证集"""

    # train_db, val_db = torch.utils.data.random_split(train_data, [700, 180])
    print('train:', len(train_db), 'validation:', len(val_db))

    train_loader = DataLoader(dataset=train_db, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_db, batch_size=8, shuffle=True)

    print("Train data load success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    early_stopping = EarlyStopping(patience=80, verbose=True)
    model.to(device)
    result_list = []
    epochs = 80
    running_loss = 0
    print("start training epoch {}".format(epochs))
    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs1, inputs2, labels1, labels2, labels3, labels4 = data
            inputs1, inputs2, labels1, labels2, labels3, labels4 = inputs1.to(device), inputs2.to(device), \
                                                          labels1.to(device), labels2.to(device), labels3.to(device), labels4.to(device)
            optimizer.zero_grad()
            logps1, logps2, embedding1, embedding2, logps_s = model.forward(inputs1, inputs2)
            loss1 = criterion(logps1, labels1)
            print("loss1", loss1)

            loss2 = criterion(logps2, labels2)

            print("loss2", loss2)

            euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
            # euclidean_distance = F.cosine_similarity(embedding1, embedding2)
            loss_contrastive = torch.mean((1 - labels3.float()) * torch.pow(euclidean_distance.float(), 2) +
                                          (labels3.float()) * torch.pow(torch.clamp(2.0 - euclidean_distance.float(), min=0.0), 2))
            print("loss3", loss_contrastive)

            loss4 = criterion(logps_s, labels4)
            print("loss4", loss4)

            loss3 = 0.5*loss1 + 0.5*loss2
            loss = 0.7*loss3 + 0.1*loss_contrastive + 0.2*loss4
            print("loss", loss)

            _, predict1 = torch.max(logps1, 1)
            _, predict2 = torch.max(logps2, 1)
            _, predict_s = torch.max(logps_s, 1)
            print("train ground truth", labels1)
            print("train predict", predict1)
            print("train ground truth", labels2)
            print("train predict", predict2)
            print("pair label", labels3)
            print("pair euclidean_distance", euclidean_distance)
            print("score label", labels4)
            print("score predict", predict_s)

            correct1 += (predict1 == labels1).sum().item()
            total1 += labels1.size(0)
            correct2 += (predict2 == labels2).sum().item()
            total2 += labels2.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, running_loss / len(train_loader)))
        print("The accuracy of total {} images: {}%".format(total1, 100 * correct1 / total1))
        print("The accuracy of total {} images: {}%".format(total2, 100 * correct2 / total2))
        running_loss = 0
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        classnum = 2
        target_num = torch.zeros((1, classnum))
        predict_num = torch.zeros((1, classnum))
        acc_num = torch.zeros((1, classnum))
        model.eval()
        with torch.no_grad():
            print("validation...")
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                inputs1, inputs2, labels1, labels2, labels3, labels4 = data
                inputs1, inputs2, labels1, labels2, labels3, labels4 = inputs1.to(device), inputs2.to(device), \
                                                                       labels1.to(device), labels2.to(
                    device), labels3.to(device), labels4.to(device)
                output1, output2, embedding1, embedding2, logps_s = model.forward(inputs1, inputs2)
                _, predict1 = torch.max(output1, 1)
                print("output1", output1)
                _, predict2 = torch.max(output2, 1)
                loss1 = criterion(output1, labels1)
                loss2 = criterion(output2, labels2)
                print("loss1", loss1)
                print("loss2", loss2)
                euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
                loss_contrastive = torch.mean((1 - labels3.float()) * torch.pow(euclidean_distance.float(), 2) +
                                              (labels3.float()) * torch.pow(
                    torch.clamp(2.0 - euclidean_distance.float(), min=0.0),
                    2))

                loss4 = criterion(logps_s, labels4)
                print("loss4", loss4)

                print("loss3", loss_contrastive)
                loss = torch.add(loss1, loss2)
                loss = torch.add(loss, loss_contrastive)
                print("loss", loss)

                running_loss += loss.item()
                total1 += labels1.size(0)
                total2 += labels2.size(0)
                print("valid ground truth", labels1)
                print("valid predict", predict1)
                print("valid ground truth", labels2)
                print("valid predict", predict2)
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
            recall = acc_num / target_num
            precision = acc_num / predict_num
            F1 = 2 * recall * precision / (recall + precision)
            # 精度调整
            recall = (recall.numpy()[0] * 100).round(3)
            precision = (precision.numpy()[0] * 100).round(3)
            F1 = (F1.numpy()[0] * 100).round(3)

            print("The accuracy of valid {} images: {}%".format(total1, 100 * correct1 / total1))
            print(
                "The accuracy of valid {} images: recall {}, precision {}, F1 {}".format(total1, recall, precision, F1))
            result_list.append([epoch, train_loss, val_loss, correct1 / total1, correct2 / total2,  recall[1], precision[1], F1[1]])

            # 输入日志
            name = ['epoch', 'train_loss', 'val_loss', 'val_acc1', 'val_acc2', 'recall', 'precision', 'F1']
            result = pd.DataFrame(columns=name, data=result_list)
            # result.to_csv("/data1/qiaohezhe/ADNI/log/ad_vs_mci/c3d_mri_epoch200_bn24_baseline_rate.csv", mode='w', index=False,
            #               header=False)
            # torch.save(model, "/data1/qiaohezhe/ADNI/log/ad_vs_mci/model/c3d/c3d_mri_epoch200_bn24_{}.pth".format(epoch))

            result.to_csv("/data1/qiaohezhe/MRI_Score/log/ad_nc/c3d_compare_score.csv", mode='w', index=False,
                          header=False)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_score/c3d_score.pth")
                print("Early stopping")
                break
    #         torch.save(model,
    #                    "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_score/c3d_score{}.pth".format(epoch))
    # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_score/c3d_score.pth")

    stop = datetime.now()
    print("Running time: ", stop - start)
