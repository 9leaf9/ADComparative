# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/10/12
# version： Python 3.7.8
# @File : inference_mri.py
# @Software: PyCharm
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
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime
from ADNI_infer import AdniDataSet
from PIL import Image
import cv2
import nibabel
from scipy import ndimage

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2]))

start = datetime.now()


def drop_invalid_range(volume, label=None):
    """
    Cut off the invalid area
    """
    zero_value = volume[0, 0, 0]
    non_zeros_idx = np.where(volume != zero_value)

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

    if label is not None:
        return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
    else:
        return volume[min_z:max_z, min_h:max_h, min_w:max_w]


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
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(1, 1, kernel_size=(6, 3, 3)),
            nn.ReLU(),
            nn.ConvTranspose1d(1, 1, kernel_size=(6, 3, 3)),
            nn.AvgPool3d((1, 8, 8)),
            nn.Sigmoid()
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
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(1, 1, kernel_size=(6, 3, 3)),
            nn.ReLU(),
            nn.ConvTranspose1d(1, 1, kernel_size=(6, 3, 3)),
            nn.AvgPool3d((1, 8, 8)),
            nn.Sigmoid()
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
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(1, 1, kernel_size=(6, 3, 3)),
            nn.ReLU(),
            nn.ConvTranspose1d(1, 1, kernel_size=(6, 3, 3)),
            nn.AvgPool3d((1, 8, 8)),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1_sw1 = self.sw1(x1)
        x1_sw2 = self.sw2(x1.transpose(2, 3))
        x1_sw3 = self.sw3(x1.transpose(2, 4))

        x1 = x1 * x1_sw1
        x1 = x1 * x1_sw2.transpose(2, 3)
        x1 = x1 * x1_sw3.transpose(2, 4)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.avgpool(x1)
        x1 = self.fc(x1.view(x1.shape[0], -1))
        x1 = self.fc2(x1)
        x1 = self.fc3(x1)
        return x1


class GuidedPropo():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None  # 需要输出的热力图
        self.relu_forward_output = []  # 记录前行传播过程中，ReLU层的输出
        self._hook_layers()

    def _hook_layers(self):
        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            在前向传播时，将ReLU层的输出保存起来
            """
            self.relu_forward_output.append(ten_out)

        def relu_backward_hook_function(module, grad_in, grad_out):
            last_relu_output = self.relu_forward_output[-1]
            # 正向传播时，ReLU输出大于0的位置设置为1， 小于0的位置设置为0
            # 反向传播时，使用这个mask对出入的梯度进行设置，满足guided propagation算法
            mask = last_relu_output[last_relu_output > 0] = 1
            # 输入梯度小于0的位置设置为0
            modified_grad_in = torch.clamp(grad_in[0], min=0.0)
            # 最终的输出梯度
            modified_grad_out = modified_grad_in * mask
            # 再次向后传播梯度时，要更新最后一层的ReLU
            del self.relu_forward_output[-1]
            # 返回值与grad_out类型相同，都是tuple
            return (modified_grad_out,)

        def conv_backward_hook_function(module, grad_in, grad_out):
            """
            反向传播到第一层卷积层时，输出的梯度就是我们需要的热力图
            """
            self.gradients = grad_in[0]

        for index, module in enumerate(list(self.model.modules()), 1):
            print(index)
            print(module)
            # if isinstance(module, nn.Conv3d) and index == 113:  # 第一层卷积层
            if isinstance(module, nn.Conv3d):
                module.register_backward_hook(conv_backward_hook_function)
            elif isinstance(module, nn.ReLU):
                module.register_forward_hook(relu_forward_hook_function)
                module.register_backward_hook(relu_backward_hook_function)

    def generate_cam(self, input_image, input_mask, target_class):
        # Forward pass.
        input_image = Variable(input_image, requires_grad=True)
        input_mask = Variable(input_mask, requires_grad=True)
        model_out = self.model(input_image, input_mask)  # shape [1, 1000]
        # Target grad.
        one_hot_output = torch.zeros(size=(1, model_out.size(1)), dtype=torch.float32)  # shape [1, 1000]
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(device)
        # Backward pass.
        model_out.backward(gradient=one_hot_output)
        # self.gradients.shape = [1, 3, 224, 224]
        image_as_array = self.gradients.data.cpu().numpy()[0]
        return image_as_array


class GradExtractor():
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None

    def _save_grad(self, grad):
        self.gradients = grad

    def _forward_pass_on_convolution_layer(self, x1, x2):
        conv_output = None

        module = list(self.model.children())
        index = 0
        for model in module:
            print(index)
            print(model)
            index += 1

        x1 = module[0](x1)
        x1.register_hook(self._save_grad)
        conv_output = x1
        x1 = module[1](x1)

        x1 = module[2](x1)

        x1 = module[3](x1)
        x_score1 = x1
        x1 = module[6](x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = module[7](x1)
        x1 = module[8](x1)
        x1 = module[9](x1)

        x2 = module[0](x2)
        x2 = module[1](x2)
        x2 = module[2](x2)
        x2 = module[3](x2)
        x2 = module[6](x2)
        x_score2 = x2
        x2 = x2.view(x2.size(0), -1)
        x2 = module[7](x2)
        x2 = module[8](x2)
        x2 = module[9](x2)

        x_score = x_score1 - x_score2
        x_score = x_score.view(x_score.shape[0], -1)
        x_score = module[10](x_score)
        x_score = module[11](x_score)


        return x1, conv_output

        # for index, module in enumerate(self.model.children(), 1):
        #     print(x1.shape)
        #     print(index)
        #     print(module)
        #     x1 = module(x1)
        #     if index == 4:
        #         x1 = x1.view(x1.size(0), -1)
        #     if index == self.target_layer:
        #         # 反向传播时调用register_hook中注册的函数
        #         x.register_hook(self._save_grad)
        #         conv_output = x
        # return conv_output, x  # 指定卷积层的输出和model.features的输出

    def forward_pass(self, x1, x2):
        output, conv_output= self._forward_pass_on_convolution_layer(x1, x2)
        # output = output.view(output.size(0), -1)  # Flatten
        # model_output = self.model.classifier(output)
        # return model_output, conv_output
        return output, conv_output


# grad cam算法实现
class GradCam():
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.extractor = GradExtractor(model, target_layer)

    def generate_cam(self, inputs1, inputs2, target_class=None):
        model_output, conv_output = self.extractor.forward_pass(inputs1, inputs2)
        print(model_output)
        if target_class == None:
            # target_class = torch.argmax(model_output.cpu(), dim=1).data.numpy()
            _, target_class = torch.max(model_output, 1)
        # Target for backprop
        print("target_class", target_class)
        one_hot_output = torch.zeros(size=(1, model_output.size(1)))
        one_hot_output[0][target_class] = 1.
        one_hot_output = one_hot_output.to(device)
        # Zero grad.
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gridient
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]  # 卷积层的输出(256, 13, 13)
        # Get weights from gradients, Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2, 3))  # 取每个gradient的均值作为weight
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :, :]
        cam = np.maximum(cam, 0)  # 相当于ReLU,小于0的值置为0
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        # 上采样到与原图片一样的大小
        # cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3], input_image.shape[4]), Image.BILINEAR))
        return cam


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path1 = "/data1/qiaohezhe/MRI_Score/ad_nc2/val/0.nii"
    img_path2 = "/data1/qiaohezhe/MRI_Score/ad_nc2/val/0.nii"
    model_path = "/data1/qiaohezhe/MRI_Score/log/ad_nc_all/model1/group/c3d_score820_score4/c3d_score33.pth"
    # model = FirstNet(f=4)
    # model = nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load(model_path))

    # model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes())
    model = torch.load(model_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(device)
    # print(model)
    # guided_cam = GuidedPropo(model)

    train_data = AdniDataSet(img_path1, img_path2)
    inputs1, inputs2 = train_data.get_mri()
    inputs1 = torch.from_numpy(inputs1).unsqueeze(dim=0)
    inputs2 = torch.from_numpy(inputs2).unsqueeze(dim=0)
    inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
    # logps1, logps2, embedding1, embedding2, logps_s, logps_score1, logps_score2 = model(inputs1, inputs2)
    # output1 = torch.softmax(torch.transpose(logps1, 0, 1), dim=0)

    # output = model(inputs)
    # image_as_array = guided_cam.generate_cam(inputs, mask, 1)
    # print(image_as_array.shape)
    #
    # for i in range(127):
    #     image = image_as_array[3, i, :, :]
    #     np.maximum(image, 0)
    #     image_for_show = (image - np.min(image)) / (np.max(image) - np.min(image))
    #     # print(image_as_array[i, :, :, :])
    #     image_for_show = np.uint8(255 * image_for_show)  # 将热力图转换为RGB格式
    #     image_for_show = cv2.applyColorMap(image_for_show, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    #     cv2.imwrite('/data1/qiaohezhe/ADNI/infer/image/out{}.jpg'.format(i), image_for_show)

    # grad_extractor = GradExtractor(model, 11)
    # input_image_var = Variable(inputs, requires_grad=True)
    # model_output, conv_output = grad_extractor.forward_pass(input_image_var)
    # one_hot_output = torch.ones(size=(1, model_output.size(1)), dtype=torch.float32)
    # one_hot_output[0][243] = 1.
    # model_output.backward(gradient=one_hot_output)
    # conv_grad = grad_extractor.gradients
    # print(model_output.shape, conv_output.shape, conv_grad.shape)

    inputs1 = Variable(inputs1, requires_grad=True)
    inputs2 = Variable(inputs2, requires_grad=True)
    grad_cam = GradCam(model, 1)
    cam = grad_cam.generate_cam(inputs1, inputs2)
    [depth, height, width] = cam.shape
    scale = [256 * 1.0 / depth, 256 * 1.0 / height, 256 * 1.0 / width]
    cam = ndimage.interpolation.zoom(cam, scale, order=0)
    print(cam.shape)
    img = nibabel.load(img_path1)
    img = img.get_data()
    print(img.shape)
    # img = drop_invalid_range(img)
    [depth, height, width] = img.shape
    scale = [256 * 1.0 / depth, 256 * 1.0 / height, 256 * 1.0 / width]
    img = ndimage.interpolation.zoom(img, scale, order=0)

    for i in range(180):
        image = cam[i, :, :]
        # image_for_show = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image_for_show = np.uint8(255 * image_for_show)  # 将热力图转换为RGB格式
        image_for_show = np.uint8(image)
        mri = img[:, :, i]

        mri = np.uint8(mri)  # 将热力图转换为RGB格式
        mri = np.expand_dims(mri, axis=2)
        mri = np.concatenate((mri, mri, mri), axis=-1)
        image_for_show = cv2.applyColorMap(image_for_show, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = image_for_show * 0.2 + mri * 1  # 这里的0.4是热力图强度因子
        # new_image_for_show = np.zeros((512, 512, 3))
        # for j in range(3):
        #     new_image_for_show[:, :, j] = cv2.resize(image_for_show[:, :, j], (256, 180))
        # print(new_image_for_show.shape)
        cv2.imwrite('/data1/qiaohezhe/MRI_Score/infer/x_slice/image/out{}.jpg'.format(i), superimposed_img)
        cv2.imwrite('/data1/qiaohezhe/MRI_Score/infer/x_slice/cam/out{}.jpg'.format(i), image_for_show)
        # cv2.imwrite('/data1/qiaohezhe/MRI_Score/infer/z_slice/mri/out{}.jpg'.format(i), img_mri)

    # cam_gb = np.multiply(cam, image_as_array)
    # # 图像归一化
    # print(cam_gb.shape)
    # cam_gb = np.maximum(cam_gb, 0)
    # # cam_gb = (cam - np.min(cam_gb)) / (np.max(cam_gb) - np.min(cam_gb))
    # plt.imshow(np.transpose(cam_gb, (1, 2, 0)))
    # plt.show()
