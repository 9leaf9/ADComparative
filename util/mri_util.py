# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/26
# version： Python 3.7.8
# @File : mri_util.py
# @Software: PyCharm

from deepbrain import Extractor
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from util.image_util import *

def skull_stripper(image):
    ext = Extractor()
    prob = ext.run(image)
    # set the threshold for extractor to extract the brain
    mask = prob < 0.7
    img_filtered = np.ma.masked_array(image, mask=mask)
    # fill the background to 0
    img_filtered = img_filtered.filled(0)

    return img_filtered


if __name__ == "__main__":
    example_filename = '../MRI/002_S_0619/lastest/001.nii'
    img = nib.load(example_filename)
    iskull = skull_stripper(np.array(img.get_data()))
    # plt.imshow(iskull[128], aspect=0.5, extent=[0, 200, 0, 400], cmap="gray")
    # plt.show()
    new_image = nib.Nifti1Image(iskull, np.eye(4))
    showNii(new_image)