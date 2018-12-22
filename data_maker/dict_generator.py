# -*- coding: utf-8 -*-

import os
import glob
import scipy.io as sio
import shutil
import numpy as np
import pandas as pd
import utils
from PIL import Image


def split_300w():
    # フォルダを生成
    train_path = '../dataset/divided/train/'
    os.makedirs(train_path, exist_ok=True)
    valid_path = '../dataset/divided/valid/'
    os.makedirs(valid_path, exist_ok=True)

    # 10度づつに分割
    degree = -90
    folder_name = ''
    train_dict = {}
    valid_dict = {}
    while degree < 90:
        folder_name = degree
        # train folder
        train_dir = os.path.join(train_path, str(folder_name))
        train_dict[degree] = train_dir
        os.makedirs(train_dir, exist_ok=True)

        # validation folder
        valid_dir = os.path.join(valid_path, str(folder_name))
        valid_dict[degree] = valid_dir
        os.makedirs(valid_dir, exist_ok=True)

        degree += 10

    """
    dir_path_ls = ['AFW', 'AFW_Flip',
                   'HELEN', 'HELEN_Flip',
                   'IBUG', 'IBUG_Flip',
                   'LFPW', 'LFPW_Flip']
    """
    degree_th = 10
    dir_path_ls = ['AFW']
    dataset_path = '../../dataset/300W_LP'

    for each_dir in dir_path_ls:
        dir_path = os.path.join(dataset_path, each_dir)
        jpg_images = glob.glob(dir_path+'/*.jpg')

        for jpg_imgae in jpg_images:
            mat_file = utils.get_matpath(jpg_imgae)
            pitch, yaw, roll = utils.get_degree_from_mat(mat_file)

            if abs(pitch) <= degree_th and abs(roll) <= degree_th:
                # ファイル名を取得
                file_name = os.path.basename(jpg_imgae)
                random_dir = np.random.choice(
                    # 20%
                    [train_dict[int(yaw - yaw % 10)],
                     valid_dict[int(yaw - yaw % 10)]],
                    p=[0.8, 0.2]
                )
                save_path = os.path.join(random_dir, file_name)
                img = utils.crop_image(mat_file, jpg_imgae)
                img.save(save_path)


def split_AFLW(test_path, dataset_path):
    """
    指定した角度ごとにフォルダを分割
    """
    # フォルダを生成 10度づつに分割
    os.makedirs(test_path, exist_ok=True)
    degree = -90
    folder_name = ''
    test_dict = {}
    while degree < 90:
        folder_name = degree
        test_dir = os.path.join(test_path, str(folder_name))
        test_dict[degree] = test_dir
        os.makedirs(test_dir, exist_ok=True)

        degree += 10

    degree_th = 10
    jpg_images = glob.glob(dataset_path + '/*.jpg')

    for jpg_imgae in jpg_images:
        mat_file = utils.get_matpath(jpg_imgae)
        pitch, yaw, roll = utils.get_degree_from_mat(mat_file)

        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            # ファイル名を取得
            file_name = os.path.basename(jpg_imgae)
            save_path = os.path.join(
                test_dict[int(yaw - yaw % 10)],
                file_name
            )
            img = utils.crop_image(mat_file, jpg_imgae)
            img.save(save_path)


if __name__ == '__main__':
    """
    split_AFLW(
        test_path = '../dataset/divided/test/',
        dataset_path = '../../dataset/AFLW2000')
    """
    split_300w()
