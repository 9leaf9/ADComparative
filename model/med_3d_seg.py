# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/9/26
# version： Python 3.7.8
# @File : med_3d_seg.py

# coding=utf-8
import os
import sys

sys.path.append('/home/qiaohezhe/code/MRI_Score/')
from setting_med import parse_opts
from model import generate_model
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
import time
from scipy import ndimage
import torch.nn.functional as F
from util.logger import log
from datasets.ADNI_spilt import AdniDataSet
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [6]))


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


def train(train_loader, val_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(train_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = nn.CrossEntropyLoss()
    print("Current setting is:")
    if not sets.no_cuda:
        criterion = criterion.cuda()

    model.train()
    result_list = []
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))
        running_loss = 0
        total = 0
        correct = 0
        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            volumes, label_masks = batch_data

            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()
            out_masks = model(volumes)

            probs = F.softmax(out_masks, dim=1)
            _, predict = torch.max(probs, 1)
            print("train ground truth", label_masks)
            print("train predict", predict)
            # calculating loss
            label_masks = label_masks.cuda()
            correct += (predict == label_masks).sum().item()
            total += label_masks.size(0)

            loss_value_seg = criterion(out_masks, label_masks)
            loss = loss_value_seg
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, total_epochs, running_loss / len(train_loader)))
        print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))

        # 模型验证
        total = 0
        correct = 0
        model.eval()
        classnum = 2
        target_num = torch.zeros((1, classnum))
        predict_num = torch.zeros((1, classnum))
        acc_num = torch.zeros((1, classnum))
        with torch.no_grad():
            print("validation...")
            for val_id, val_data in enumerate(val_loader):
                # forward
                val_volume, labels = val_data
                if not sets.no_cuda:
                    val_volume = val_volume.cuda()
                with torch.no_grad():
                    probs = model(val_volume)
                    output = F.softmax(probs, dim=1)
                    _, predict = torch.max(output, 1)
                    total += labels.size(0)
                    correct += (predict.cpu() == labels).sum().item()
                    print("valid ground truth", labels)
                    print("valid predict", predict)
                    '''calculate  Recall Precision F1'''
                    pre_mask = torch.zeros(output.size()).scatter_(1, predict.cpu().view(-1, 1), 1.)
                    predict_num += pre_mask.sum(0)
                    tar_mask = torch.zeros(output.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
                    target_num += tar_mask.sum(0)
                    acc_mask = pre_mask * tar_mask
                    acc_num += acc_mask.sum(0)

            val_loss = running_loss / len(train_loader)
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

            result.to_csv("/data1/qiaohezhe/MRI_Score/log/ad_nc_all/paper1/methods/med_baseline1.csv", mode='w',
                          index=False,
                          header=False)

            early_stopping(val_loss, model)


if __name__ == '__main__':
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
    val_data_path1 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized.csv'
    val_data_path2 = '/data1/qiaohezhe/miriad/miriad_val_label_normalized_shuffle.csv'

    # val_data_path1 = '/data1/qiaohezhe/AD2/adni2_ad_nc_val_label.csv'
    # val_data_path2 = '/data1/qiaohezhe/AD2/adni2_ad_nc_val_label_shuffle.csv'

    train_img_path = '/data1/qiaohezhe/MRI_Score/ad_nc/train/'
    val_img_path = '/data1/qiaohezhe/miriad/ad_nc_regsiter/'

    train_db = AdniDataSet(train_data_path1, train_img_path, sets)
    val_db = AdniDataSet(val_data_path1, val_img_path, sets)

    """将训练集划分为训练集和验证集"""

    train_loader = DataLoader(dataset=train_db, batch_size=12, shuffle=True)
    val_loader = DataLoader(dataset=val_db, batch_size=12, shuffle=True)

    # getting models
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    print(model)
    # optimizer
    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
            {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
        ]
    early_stopping = EarlyStopping(patience=20, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # train from resume
    # if sets.resume_path:
    #     if os.path.isfile(sets.resume_path):
    #         print("=> loading checkpoint '{}'".format(sets.resume_path))
    #         checkpoint = torch.load(sets.resume_path)
    #         models.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(sets.resume_path, checkpoint['epoch']))

    # training
    train(train_loader, val_loader, model, optimizer, scheduler, total_epochs=100,
          save_interval=sets.save_intervals,
          save_folder=sets.save_folder, sets=sets)
