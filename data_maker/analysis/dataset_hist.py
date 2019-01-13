# -*- coding: utf-8 -*-

import os
import glob
import scipy.io as sio
import matplotlib.pyplot as plt
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


pose_ls = []
df_pose = None
dataset_path = '../../../dataset/300W_LP'
dir_path_ls = ['AFW', 'AFW_Flip',
               'HELEN', 'HELEN_Flip',
               'IBUG', 'IBUG_Flip',
               'LFPW', 'LFPW_Flip']
csv_file_name = 'df_300w_lp.csv'
csv_path = './'+ csv_file_name
"""
dataset_path = '../../dataset/AFLW2000'
dir_path_ls = ['.']
"""
# CSVファイルの存在の確認
if os.path.exists(csv_path):
    df_pose = pd.read_csv(csv_path)
else:
    for each_dir in dir_path_ls:
        dir_path = os.path.join(dataset_path, each_dir)
        files = glob.glob(dir_path+'/*.mat')

        for file in files:
            pose_params = get_ypr_from_mat(file)
            pose_ls.append(pose_params)

    # 度数に変換
    pitch = [pose[0] * 180 / np.pi for pose in pose_ls]
    yaw = [pose[1] * 180 / np.pi for pose in pose_ls]
    roll = [pose[2] * 180 / np.pi for pose in pose_ls]

    df_pose = pd.DataFrame(
        {"pitch": pitch, 'yaw': yaw, 'roll': roll}
    )

    # CSVに出力
    df_pose.to_csv(csv_file_name)

# YAW
# 絶対値10度以上のやつを除外
yaw_pose = (np.fabs(df_pose['pitch']) <= 10) & (
    np.fabs(df_pose['roll']) <= 10)
df_pose = df_pose[yaw_pose]

df_pose['yaw'].hist(bins=18)
plt.xlabel("degree")
plt.ylabel('number')
plt.xticks(np.arange(-90, 90, 10))
plt.savefig('yaw_hist.png')

# ROLL
# 絶対値10度以上のやつを除外
plt.figure()
roll_pose = (np.fabs(df_pose['yaw']) <= 10) & (
    np.fabs(df_pose['pitch']) <= 10)
df_pose = df_pose[roll_pose]

df_pose['roll'].hist(bins=18)
plt.xlabel("degree")
plt.ylabel('number')
plt.xticks(np.arange(-90, 90, 10))
plt.savefig('roll_hist.png')

# PITCH
# 絶対値10度以上のやつを除外
plt.figure()
pitch_pose = (np.fabs(df_pose['yaw']) <= 10) & (
    np.fabs(df_pose['roll']) <= 10)
df_pose = df_pose[pitch_pose]

df_pose['pitch'].hist(bins=18)
plt.xlabel("degree")
plt.ylabel('number')
plt.xticks(np.arange(-90, 90, 10))
plt.savefig('pitch_hist.png')
