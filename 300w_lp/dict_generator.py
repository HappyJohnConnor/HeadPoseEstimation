# -*- coding: utf-8 -*-

import os
import glob
import scipy.io as sio
import shutil
import numpy as np
import pandas as pd
from PIL import Image


def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d


def crop_and_save(mat_path, img_path, save_path):
    img = Image.open(img_path)
    pt2d = get_pt2d_from_mat(mat_path)
    x_min = min(pt2d[0, :])
    y_min = min(pt2d[1, :])
    x_max = max(pt2d[0, :])
    y_max = max(pt2d[1, :])

    # k = 0.2 to 0.40
    k = np.random.random_sample() * 0.2 + 0.2
    x_min -= 0.6 * k * abs(x_max - x_min)
    y_min -= 2 * k * abs(y_max - y_min)
    x_max += 0.6 * k * abs(x_max - x_min)
    y_max += 0.6 * k * abs(y_max - y_min)
    img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

    # save iamges
    img.save(save_path)


# フォルダを生成
BASE_DIR = os.path.realpath(os.path.dirname(__file__))

# 10度づつに分割
degree = -90
folder_name = ''
folder_dict = {}
while degree < 90:
    folder_name = degree
    DEGREE_DIR = os.path.join(BASE_DIR, str(folder_name))
    if not os.path.exists(DEGREE_DIR):
        os.mkdir(DEGREE_DIR)
    folder_dict[degree] = DEGREE_DIR
    degree += 10


#dir_path_ls = ['AFW', 'HELEN', 'IBUG', 'LFPW']
files_ls = []
degree_th = 10
dir_path_ls = ['AFW']
for each_dir in dir_path_ls:
    dir_path = os.path.join('../300W_LP', each_dir)
    mat_files = glob.glob(dir_path+'/*.mat')
    jpg_images = glob.glob(dir_path+'/*.jpg')

    for mat_file, jpg_imgae in zip(mat_files, jpg_images):
        pose_params = get_ypr_from_mat(mat_file)

        # 度数に変換
        pitch = pose_params[0] * 180 / np.pi
        yaw = pose_params[1] * 180 / np.pi
        roll = pose_params[2] * 180 / np.pi

        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            # ファイル名を取得
            file_name = os.path.basename(jpg_imgae)
            save_path = os.path.join(
                folder_dict[int(yaw - yaw % 10)], file_name)
            crop_and_save(mat_file, jpg_imgae, save_path)
