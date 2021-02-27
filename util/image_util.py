# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/6
# version： Python 3.7.8
# @File : image_util.py
# @Software: PyCharm

# 处理nii类型图片
from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import numpy as np
import matplotlib


# matplotlib.use('TkAgg')
# 文件名，nii或nii.gz
def showimg(path):
    img = nib.load(example_filename)

    # 打印文件信息
    print(img)
    print(img.dataobj.shape)
    print(img.get_data().max())
    width, height, queue = img.dataobj.shape
    # print(type(img.dataobj))
    # 显示3D图像
    OrthoSlicer3D(img.dataobj).show()

    # 计算看需要多少个位置来放切片图
    x = int((queue / 10) ** 0.5) + 1
    y = int((height / 10) ** 0.5) + 1
    z = int((width / 10) ** 0.5) + 1
    num = 1
    # 按照10的步长，切片，显示2D图像
    # for i in range(0, queue, 10):
    #     img_arr = img.dataobj[:, :, i]
    #     img_arr = np.squeeze(img_arr)
    #     plt.subplot(x, x, num)
    #     plt.imshow(img_arr, cmap='gray')
    #     num += 1

    # 按照10的步长，切片，显示2D图像
    # for i in range(0, height, 10):
    #     img_arr = img.dataobj[:, i, :]
    #     img_arr = np.squeeze(img_arr)
    #     plt.subplot(y, y, num)
    #     plt.imshow(img_arr, cmap='gray')
    #     num += 1

    # 按照10的步长，切片，显示2D图像
    for i in range(0, width, 10):
        img_arr = img.dataobj[i, :, :]
        # img_arr = np.squeeze(img_arr)
        plt.subplot(z, z, num)
        plt.imshow(img_arr, cmap='gray')
        num += 1

    plt.show()


def showNii(img):
    # 打印文件信息
    print(img)
    print(img.dataobj.shape)
    print(img.get_data().max())
    width, height, queue = img.dataobj.shape
    # print(type(img.dataobj))
    # 显示3D图像
    OrthoSlicer3D(img.dataobj).show()

    # 计算看需要多少个位置来放切片图
    x = int((queue / 10) ** 0.5) + 1
    y = int((height / 10) ** 0.5) + 1
    z = int((width / 10) ** 0.5) + 1
    num = 1
    # 按照10的步长，切片，显示2D图像
    for i in range(0, queue, 10):
        img_arr1 = img.dataobj[:, :, i]
        img_arr1 = np.squeeze(img_arr1)
        plt.subplot(x, x, num)
        plt.imshow(img_arr1, cmap='gray')
        num += 1
    num = 1
    # 按照10的步长，切片，显示2D图像
    for i in range(0, height, 10):
        img_arr2 = img.dataobj[:, i, :]
        img_arr2 = np.squeeze(img_arr2)
        plt.subplot(y, y, num)
        plt.imshow(img_arr2, cmap='gray')
        num += 1
    num = 1
    # 按照10的步长，切片，显示2D图像
    for i in range(0, width, 10):
        img_arr3 = img.dataobj[i, :, :]
        img_arr3 = np.squeeze(img_arr3)
        plt.subplot(z, z, num)
        plt.imshow(img_arr3, cmap='gray')
        num += 1

    plt.show()


if __name__ == "__main__":
    example_filename = '../MRI/002_S_0295/lastest/001_processed.nii'
    # example_filename = '../MRI/002_S_0295/lastest/ADNI_002_S_0295_MR_MT1__N3m_Br_20120605092714994_S150055_I308078.nii'
    # example_filename = '../data/MRBrainS18/labels/1.nii.gz'
    # example_filename = '../HFH/Label_Nii/Label_Nii_002.nii'
    # example_filename = '../HFH/predict/1_arr.nii.gz'
    showimg(example_filename)

# data = img.get_data()
# data[data > 1] = 1
# import nibabel as nib
# import numpy as np
#
# new_image = nib.Nifti1Image(data, np.eye(4))
# nib.save(new_image, '../HFH/my_arr.nii.gz')
