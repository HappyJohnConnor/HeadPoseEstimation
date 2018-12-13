# -*- coding: utf-8 -*-

import os
import glob
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


def convert_img_to_array(img):
    # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
    # [R,G,B]はそれぞれが0-255の配列。
    image = np.array(img.resize((25, 25)))
    # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
    image = image.transpose(2, 0, 1)
    # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
    image = image.reshape(
        1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]

    return image


# 学習用のデータを作る.
image_list = []
label_list = []

BASE_DIR = './300W_LP'
#dir_path_ls = ['AFW', 'HELEN', 'IBUG', 'LFPW']
files_ls = []
degree_th = 10
dir_path_ls = ['AFW']
for each_dir in dir_path_ls:
    dir_path = os.path.join(BASE_DIR, each_dir)
    mat_files = glob.glob(dir_path+'/*.mat')
    jpg_images = glob.glob(dir_path+'/*.jpg')

    for mat_file, jpg_imgae in zip(mat_files, jpg_images):
        pose_params = get_ypr_from_mat(mat_file)

        # 度数に変換
        pitch = pose_params[0] * 180 / np.pi
        yaw = pose_params[1] * 180 / np.pi
        roll = pose_params[2] * 180 / np.pi

        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            # labelに格納
            label_list.append(int(yaw - yaw % 10))
            # ファイル名を取得
            file_name = os.path.basename(jpg_imgae)
            img = crop_image(mat_file, jpg_imgae)
            # array型に変換
            img = convert_img_to_array(img)
            # 出来上がった配列をimage_listに追加。
            image_list.append(img / 255.)

print(len(label_list))
print(len(image_list))
