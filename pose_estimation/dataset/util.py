# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
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

def convert_img_to_array(img, img_size):
    # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
    # [R,G,B]はそれぞれが0-255の配列。
    image = np.array(img.resize(img_size))
    # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
    return image