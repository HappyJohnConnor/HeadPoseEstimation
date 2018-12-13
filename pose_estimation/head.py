# -*- coding: utf-8 -*-

import glob
#from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from model import googlenet2
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
from dataset import util

image_list = []
label_list = []

TRAIN_DIR = './dataset/train/300W_LP'
#dir_path_ls = ['AFW', 'HELEN', 'IBUG', 'LFPW']
degree_th = 10
img_size = (224, 224)
dir_path_ls = ['AFW']
for each_dir in dir_path_ls:
    dir_path = os.path.join(TRAIN_DIR, each_dir)
    mat_files = glob.glob(dir_path+'/*.mat')
    jpg_images = glob.glob(dir_path+'/*.jpg')

    for mat_file, jpg_imgae in zip(mat_files, jpg_images):
        pose_params = util.get_ypr_from_mat(mat_file)

        # 度数に変換
        pitch = pose_params[0] * 180 / np.pi
        yaw = pose_params[1] * 180 / np.pi
        roll = pose_params[2] * 180 / np.pi

        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            # labelに格納
            # 0, 1, 2, .. となるようにする
            label_list.append(int((yaw - yaw % 10) + 90)/10)
            # ファイル名を取得
            file_name = os.path.basename(jpg_imgae)
            img = util.crop_image(mat_file, jpg_imgae)
            # array型に変換
            img = util.convert_img_to_array(img, img_size)
            # 出来上がった配列をimage_listに追加。
            image_list.append(img / 255.)

print(len(label_list))
print(len(image_list))

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
Y = to_categorical(label_list)
print(Y)
print (Y.shape)

# オプティマイザにAdamを使用
opt = Adam(lr=0.001)
# モデルをコンパイル
model = googlenet2.create_googlenet('./model/googlenet_weights.h5')
model.compile(loss="categorical_crossentropy",
                optimizer=opt, 
                metrics=["accuracy"])
# 学習を実行。10%はテストに使用。
print (image_list[0].shape)
model.fit(image_list, Y, epochs=1500,
               batch_size=10, validation_split=0.1)


"""
ここからテスト
"""
total = 0.
ok_count = 0.

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
TEST_DIR = './dataset/test/AFLW2000'
degree_th = 10
mat_files = glob.glob(TEST_DIR + '/*.mat')
jpg_images = glob.glob(TEST_DIR + '/*.jpg')

for mat_file, jpg_imgae in zip(mat_files, jpg_images):
    pose_params = util.get_ypr_from_mat(mat_file)

    # 度数に変換
    pitch = pose_params[0] * 180 / np.pi
    yaw = pose_params[1] * 180 / np.pi
    roll = pose_params[2] * 180 / np.pi

    if abs(pitch) <= degree_th and abs(roll) <= degree_th:
        image = np.array(Image.open(jpg_imgae).resize((25, 25)))
        image = image.transpose(2, 0, 1)
        image = image.reshape(
            1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
        result = model.predict_classes(np.array([image / 255.]))

        label = int(yaw - yaw % 10)
        print("label:", label, "result:", result[0])

        total += 1.

        if label == int(result[0]):
            ok_count += 1.


print("seikai: ", ok_count / total * 100, "%")

# モデルの可視化

#SVG(model_to_dot(model).create(prog='dot', format='svg'))
