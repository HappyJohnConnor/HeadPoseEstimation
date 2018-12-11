import os
import glob
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
dir_path_ls = ['AFW', 'HELEN', 'IBUG', 'LFPW']
#dir_path_ls = ['AFW']
for each_dir in dir_path_ls:
    dir_path = os.path.join('../300W_LP', each_dir) 
    files = glob.glob(dir_path+'/*.mat')
    
    for file in files:
        pose_params = get_ypr_from_mat(file)
        pose_ls.append(pose_params)

# 度数に変換
pitch = [pose[0] * 180 / np.pi for pose in pose_ls]
yaw = [pose[1] * 180 / np.pi for pose in pose_ls]
roll = [pose[2] * 180 / np.pi for pose in pose_ls]

df_pose = pd.DataFrame(
    {"pitch":pitch, 'yaw':yaw, 'roll':roll}
)
# 絶対値10度以上のやつを除外
picked_pose = (np.fabs(df_pose['pitch']) <=10) & (np.fabs(df_pose['roll']) <=10)
df_pose= df_pose[picked_pose]
print(df_pose)

df_pose['yaw'].hist(bins = 20)
plt.xlabel("degree")
plt.ylabel('number')
plt.xticks(np.arange(-90, 90, 20))
plt.savefig('fig1.png')

