# -*- coding: utf-8 -*-

import numpy as np
import os
import scipy.io as sio
from PIL import Image


def get_matpath(img_path):
    file_path, file_name = os.path.split(img_path)
    name_ls = file_name.split('.')
    new_name = name_ls[0]+'.mat'
    mat_path = file_path + '/' + new_name

    return mat_path


def get_img_name(img_path):
    file_path, file_name = os.path.split(img_path)
    name_ls = file_name.split('.')
    img_name = name_ls[0]+'.jpg'

    return img_name


def get_degree_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]

    # 度数に変換
    pitch = pose_params[0] * 180 / np.pi
    yaw = pose_params[1] * 180 / np.pi
    roll = pose_params[2] * 180 / np.pi

    return pitch, yaw, roll


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']

    return pt2d


def crop_image(mat_path, img_path):
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

    return img
