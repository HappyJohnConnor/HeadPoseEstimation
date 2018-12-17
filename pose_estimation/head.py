# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
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
new_img_ls = image_list[np.newaxis]  #  (Height, Width, Channels)  -> (1, Height, Width, Channels) 

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
Y = to_categorical(label_list)
print(Y)
print(Y.shape)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=False))

gen_flow = datagen.flow(new_img_ls, batch_size=16) 
# モデルをコンパイル
model = googlenet2.create_googlenet('./model/googlenet_weights.h5')
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.SGD(
                  lr=0.001, momentum=param['momentum']),
              metrics=["accuracy"])
# 学習を実行。10%はテストに使用。
history = model.fit_generator(
    datagen.flow(new_img_ls, batch_size=16), 
    Y, 
    epochs=1500,
    batch_size=10, 
    validation_split=0.4
)

model_save_path = './model/save'
# モデルの保存
json_string = model.to_json()
open(os.path.join(model_save_path, 'my_model.json'), 'w').write(json_string)
yaml_string = model.to_yaml()
open(os.path.join(model_save_path, 'my_model.yaml'), 'w').write(yaml_string)
print('save weights')
model.save_weights(os.path.join(model_save_path, 'my_model_weights.hdf5'))


# Plot accuracy &amp; loss

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

# plot accuracy
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
plt.savefig("acc.png")
plt.close()

# plot loss
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.savefig("loss.png")
plt.close()
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
