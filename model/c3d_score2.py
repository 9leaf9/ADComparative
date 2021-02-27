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
from datasets.ADNI_compare_score2 import AdniDataSet_train
from datasets.ADNI_compare_score2 import AdniDataSet_val
from setting import parse_opts
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import roc_curve, auc
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
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
        x1 = self.layer4(x1)
        x2 = self.layer4(x2)
        x_score1 = x1
        x_score2 = x2
        x_score = x_score1 - x_score2
        x_score = x_score.view(x_score.shape[0], -1)

        x_score = self.fc4(x_score)
        x_s = self.fc5(x_score)

        # x_score1 = x_score1.view(x_score1.shape[0], -1)
        # x_score1 = self.fc4(x_score1)
        # x_score1 = self.fc5(x_score1)
        #
        # x_score2 = x_score2.view(x_score2.shape[0], -1)
        # x_score2 = self.fc4(x_score2)
        # x_score2 = self.fc5(x_score2)
        #
        # x_score1 = x_score1.view(x_score1.shape[0], -1)
        # x_score2 = x_score2.view(x_score2.shape[0], -1)

        # x_s = F.sigmoid(x_s)
        # x_s = x_s.reshape(x_s.shape[0])

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
        x_embedding1 = x_s
        x_embedding2 = x_s

        return x1, x2, x_embedding1, x_embedding2, x_s, x_s1, x_s2


if __name__ == "__main__":
    sets = parse_opts()
    sets.gpu_id = [0]

    # data_path = '/data1/qiaohezhe/MRI_Score/new_ad_data_filter_mci.csv'
    # img_path = '/data1/qiaohezhe/MRI_Score/MRI/'

    # train_data = AdniDataSet(data_path, img_path, sets)
    """将训练集划分为训练集和验证集"""
    # train_db, val_db = torch.utils.data.random_split(train_data, [230, 101])
    # print('train:', len(train_db), 'validation:', len(val_db))

    train_data_path1 = '/data1/qiaohezhe/MRI_Score/mci_nc/train_label_single2.csv'
    train_data_path2 = '/data1/qiaohezhe/MRI_Score/mci_nc/train_label_single_shuffle2.csv'
    val_data_path1 = '/data1/qiaohezhe/MRI_Score/mci_nc/val_label.csv'
    val_data_path2 = '/data1/qiaohezhe/MRI_Score/mci_nc/val_label_shuffle.csv'
    # val_data_path1 = '/data1/qiaohezhe/AD2/adni2_mci_nc_val_label.csv'
    # val_data_path2 = '/data1/qiaohezhe/AD2/adni2_mci_nc_val_label_shuffle.csv'

    train_img_path = '/data1/qiaohezhe/MRI_Score/mci_nc/train/'
    val_img_path = '/data1/qiaohezhe/MRI_Score/mci_nc/val/'
    # val_img_path = '/data1/qiaohezhe/AD2/mci_nc_regsiter/'

    train_db = AdniDataSet_train(train_data_path1, train_data_path2, train_img_path, sets)
    val_db = AdniDataSet_val(val_data_path1, val_data_path2, val_img_path, sets)
    """将训练集划分为训练集和验证集"""

    # train_db, val_db = torch.utils.data.random_split(train_data, [700, 180])
    print('train:', len(train_db), 'validation:', len(val_db))

    train_loader = DataLoader(dataset=train_db, batch_size=12, shuffle=True)
    val_loader = DataLoader(dataset=val_db, batch_size=12, shuffle=True)

    print("Train data load success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FirstNet(f=8)
    model = torch.nn.DataParallel(model)

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

            logps1, logps2, embedding1, embedding2, logps_s, logps_score1, logps_score2 = model.forward(inputs1,
                                                                                                        inputs2)

            print(logps_s)
            print(logps_score1)
            print(logps_score2)

            loss1 = criterion(logps1, labels1)
            loss2 = criterion(logps2, labels2)
            loss_group = loss1 + loss2
            print("loss_group", loss_group)

            # euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
            # loss_contrastive = torch.mean((1 - labels3.float()) * torch.pow(euclidean_distance.float(), 2) +
            #                               (labels3.float()) * torch.pow(
            #     torch.clamp(4.0 - euclidean_distance.float(), min=0.0), 2))
            # print("loss_contrastive", loss_contrastive)

            # bceLoss = nn.BCELoss()
            # loss_compare = bceLoss(logps_s, labels4.float())
            loss_compare = criterion(logps_s, labels4)
            # loss_compare = F.mse_loss(logps_s, labels4.float())
            print("loss_compare", loss_compare)
            print(logps_score1.shape)
            loss3 = F.mse_loss(logps_score1, score1.float())
            loss4 = F.mse_loss(logps_score2, score2.float())

            loss_score = 0.5 * loss3 + 0.5 * loss4
            print("loss_score", loss_score)

            loss = 0.8 * loss_group + 0.2 * loss_compare + 0 * loss_score
            print("loss", loss)

            _, predict1 = torch.max(logps1, 1)
            _, predict2 = torch.max(logps2, 1)
            # _, predict_s = torch.max(logps_s, 1)
            # predict_s = (logps_s >= 0.5).type(torch.cuda.FloatTensor)

            print("train group ground truth", labels1)
            print("train group predict", predict1)

            print("train score ground truth1", score1)
            print("train score predict1", logps_score1)

            print("train score ground truth2", score2)
            print("train score predict2", logps_score2)

            print("train compare ground truth", labels4)
            print("train compare predict", logps_s)

            # print("train compare ground truth", labels3)
            # print("train compare predict", euclidean_distance)

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
                output1, output2, embedding1, embedding2, logps_s, logps_score1, logps_score2 = model.forward(inputs1,
                                                                                                              inputs2)

                loss1 = criterion(output1, labels1)
                loss2 = criterion(output2, labels2)
                loss_group = loss1 + loss2
                print("loss_group", loss_group)

                # bceLoss = nn.BCELoss()
                # loss_compare = bceLoss(logps_s, labels4.float())
                loss_compare = criterion(logps_s, labels4)
                # loss_compare = F.mse_loss(logps_s, labels4.float())
                print("loss_compare", loss_compare)

                loss3 = F.mse_loss(logps_score1, score1.float())
                loss4 = F.mse_loss(logps_score2, score2.float())

                loss_score = 0.5 * loss3 + 0.5 * loss4
                print("loss_score", loss_score)

                # euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
                # loss_contrastive = torch.mean((1 - labels3.float()) * torch.pow(euclidean_distance.float(), 2) +
                #                               (labels3.float()) * torch.pow(
                #     torch.clamp(2.0 - euclidean_distance.float(), min=0.0), 2))
                # print("loss_contrastive", loss_contrastive)

                loss = 0.8 * loss_group + 0.2 * loss_compare + 0 * loss_score
                print("loss", loss)

                _, predict1 = torch.max(output1, 1)
                _, predict2 = torch.max(output2, 1)
                # _, predict_s = torch.max(logps_s, 1)
                # predict_s = (logps_s >= 0.5).type(torch.cuda.FloatTensor)

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

                # print("train compare ground truth", labels3)
                # print("train compare predict", euclidean_distance)

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
                 round(correct2 / total2, 3), round(recall[1], 3), round(precision[1], 3), round(F1[1], 3), round(roc_auc, 3),
                 round(train_group_loss, 3), round(train_score_loss, 3), round(train_compare_loss, 3),
                 round(val_group_loss, 3), round(val_score_loss, 3),
                 round(val_compare_loss, 3)])

            # 输入日志
            name = ['epoch', 'train_loss', 'val_loss', 'val_acc1', 'val_acc2', 'recall', 'precision', 'F1', 'AUC',
                    'train_group_loss', 'train_score_loss', 'train_compare_loss', 'val_group_loss', 'val_score_loss',
                    'val_compare_loss']
            result = pd.DataFrame(columns=name, data=result_list)

            result.to_csv("/data1/qiaohezhe/MRI_Score/log/mci_nc/paper/c3d_score820_score0.csv", mode='w', index=False,
                          header=False)
            # result.to_csv("/data1/qiaohezhe/MRI_Score/log/mci_nc_adni2/paper1/group/c3d_score820_score3.csv", mode='w',
            #               index=False,
            #               header=True)

            # result.to_csv("/data1/qiaohezhe/MRI_Score/log/mci_nc_adni2/paper1/ablation/c3d_score820_score0_8.csv", mode='w',
            #               index=False,
            #               header=True)


            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_score8201/c3d_score.pth")
                # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc_adni2/model1/group/c3d_score820_score3/c3d_score.pth")
                print("Early stopping")
                break
    #         torch.save(model,
    #                    "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_score8201/c3d_score{}.pth".format(epoch))
    # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_score8201/c3d_score.pth")
    #         torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc_adni2/model1/group/c3d_score820_score3/c3d_score{}.pth".format(epoch))
    # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc_adni2/model1/group/c3d_score820_score3/c3d_score.pth")

    stop = datetime.now()
    print("Running time: ", stop - start)
