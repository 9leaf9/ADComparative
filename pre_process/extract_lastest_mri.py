# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/22
# version： Python 3.7.8
# @File : extract_lastest_mri.py
# @Software: PyCharm
'''对每个受试者提取最新的MRI， 并拷贝到相应的文件夹'''

filePath = '../MRI/'
file_lsit = os.listdir(filePath)
print(file_lsit)
for index, file in tqdm(enumerate(file_lsit), total=len(file_lsit)):
    if os.listdir(filePath + file):
        print("processing {}...".format(file))
        lastest_dirs = filePath + file + '/' + 'lastest/'
        if not os.path.exists(lastest_dirs):
            os.makedirs(lastest_dirs)
        source_file = filePath + file + '/'
        mri_path = ""
        mri_time = 0
        for root, dir, files in os.walk(source_file):
            for temp_file in files:
                if temp_file == '002.nii' or temp_file == '001.nii' or \
                        temp_file == '001_processed.nii':
                    continue
                mri_file = os.path.join(root, temp_file)
                # print(mri_file)
                result = mri_file.split('_')
                '''记录最新的mri路径'''
                if int(result[-3][0:8]) > mri_time and mri_file.find('Mask') == -1 and \
                        mri_file.find('MR_MPR__GradWarp__B1_Correction_Br') != -1:
                    mri_path = mri_file
                    mri_time = int(result[-3][0:8])
        # print(mri_path)
        '''文件拷贝'''
        if mri_path != "":
            shutil.copy(mri_path, lastest_dirs + '/002.nii')
            print("{}'s lastest copy is finshed...".format(file))
        else:
            print("The lastest MRI in {} is not exist".format(file))

    else:
        print("{} is empty".format(file))
