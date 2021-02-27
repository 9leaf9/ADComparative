# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/4/18
# version： Python 3.7.8
# @File : log_util.py
# @Software: PyCharm
'''训练过程中 train loss  validation acc的写入'''

import os


def log_write(epoch_list, train_loss_list, test_acc_list, log_path):
    content = ''
    for i in range(len(epoch_list)):
        content += str(epoch_list[i]) + ',' + str(train_loss_list[i]) + ',' + str(test_acc_list[i]) + '\n'

    if os.path.exists(log_path):
        with open(log_path, mode='w', encoding='utf-8') as fin:
            fin.write(content)
    else:
        with open(log_path, mode='w', encoding='utf-8') as fin:
            print(log_path + "文件创建成功！")
            fin.write(content)


def log_write2(epoch_list, train_loss_list, log_path):
    content = ''
    for i in range(len(epoch_list)):
        content += str(epoch_list[i]) + ',' + str(train_loss_list[i]) + '\n'

    if os.path.exists(log_path):
        with open(log_path, mode='w', encoding='utf-8') as fin:
            fin.write(content)
    else:
        with open(log_path, mode='w', encoding='utf-8') as fin:
            print(log_path + "文件创建成功！")
            fin.write(content)


# f = open("../HFH/loss_log.csv", "r")
# lines = f.readlines()  # 读取全部内容
# text = ""
# for line in lines:
#     # print(line)
#     str = line.split(',')
#     # print(str[0])
#     # print(str[1][7:])
#     text += str[0]+"," + str[1][7:]+ "\n"
#
# print(text)
# with open("../HFH/loss_log.csv", mode='w', encoding='utf-8') as fin:
#     fin.write(text)