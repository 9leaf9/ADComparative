# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/4/18
# version： Python 3.7.8
# @File : loss_acc_paint.py
# @Software: PyCharm
'''根据 train loss validation acc 绘图'''


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
import csv
from mpl_toolkits.axes_grid1 import host_subplot
# 实现插值的模块
from scipy import interpolate


# 绘制损失曲线
def loss_paint(path):
    # read the log file
    fp = open(path)
    plots = csv.reader(fp, delimiter=',')
    train_iterations = []
    train_loss = []

    for ln in plots:
        train_loss.append(float(ln[1]))
        train_iterations.append(int(ln[0]))

    fp.close()

    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()
    # set labels
    host.set_xlabel("iterations")
    host.set_ylabel("train loss")

    # 插值法之后的x轴值，表示从0到9间距为0.5的18个数
    new_epoch = np.arange(0, 50, 5)
    train_loss_func = interpolate.interp1d(train_iterations, train_loss, kind='cubic')
    # 利用xnew和func函数生成ynew，xnew的数量等于ynew数量
    new_train_loss = train_loss_func(new_epoch)

    # 拟合之后的平滑曲线图
    # plt.plot(xnew, ynew, 'r-')

    # plot curves
    p1, = host.plot(new_epoch, new_train_loss, '-.ro', label="training loss")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())

    # set the range of x axis of host and y axis of par1
    # LeNet
    host.set_ylim([0, 0.15])
    host.set_xlim([0, 50])

    # AlexNet
    # host.set_ylim([0.03, 0.45])
    # par1.set_ylim([0.95, 1])
    # host.set_xlim([0, 15])

    plt.draw()
    plt.show()


# 绘制损失 准确率曲线
def acc_loss_paint(path):
    # read the log file
    fp = open(path)
    plots = csv.reader(fp, delimiter=',')
    train_iterations = []
    train_loss = []
    test_iterations = []
    test_accuracy = []

    for ln in plots:
        train_loss.append(float(ln[1]))
        train_iterations.append(int(ln[0]))
        test_accuracy.append(float(ln[2]))
        test_iterations.append(int(ln[0]))

    fp.close()

    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()
    # set labels
    host.set_xlabel("iterations")
    host.set_ylabel("train loss")
    par1.set_ylabel("validation accuracy")

    # 插值法之后的x轴值，表示从0到9间距为0.5的18个数
    new_epoch = np.arange(0, 100, 5)
    train_loss_func = interpolate.interp1d(train_iterations, train_loss, kind='cubic')
    # 利用xnew和func函数生成ynew，xnew的数量等于ynew数量
    new_train_loss = train_loss_func(new_epoch)

    train_acc_func = interpolate.interp1d(train_iterations, test_accuracy, kind='cubic')
    # 利用xnew和func函数生成ynew，xnew的数量等于ynew数量
    new_train_acc = train_acc_func(new_epoch)

    # 拟合之后的平滑曲线图
    # plt.plot(xnew, ynew, 'r-')

    # plot curves
    p1, = host.plot(new_epoch, new_train_loss, '-.ro', label="training loss")
    p2, = par1.plot(new_epoch, new_train_acc, ':bo', label="validation accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # LeNet
    host.set_ylim([0, 0.1])
    par1.set_ylim([0.5, 1])
    host.set_xlim([0, 100])

    # AlexNet
    # host.set_ylim([0.03, 0.45])
    # par1.set_ylim([0.95, 1])
    # host.set_xlim([0, 15])

    plt.draw()
    plt.show()


if __name__ == "__main__":
    loss_paint('../HFH/loss_log.csv')
