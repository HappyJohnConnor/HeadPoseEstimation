import argparse
import glob
import numpy as np
import os

from matplotlib import pylab as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Activation, Dense, Dropout, Input, Flatten
from keras.preprocessing.image import img_to_array, load_img
from pathlib import Path
from PIL import Image

from data_maker import utils
import utils_for_keras

def get_matpath(img_path):
    base_path  = './dataset/mat'
    _, file_name = os.path.split(img_path)
    name_ls = file_name.split('.')
    new_name = name_ls[0]+'.mat'
    mat_path = base_path + '/' + new_name

    return mat_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--model_path', dest='model_path',
                        help='String appended to output model.', default='1', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    weight_path = './model/output/' + args.model_path + '/my_model.hdf5'
    img_size = 150
    model = utils_for_keras.get_model(weight_path=weight_path)
    
    # Pathオブジェクトを生成
    base_path = Path("./dataset/divided/valid/")
    path_ls = list(base_path.glob("**/*.jpg"))
    result_ls = []
    true_ls = []
    for jpg_path in path_ls:
        # matファイルを探す
        mat_file = get_matpath(jpg_path)
        # 角度を取得
        pitch, yaw, roll = utils.get_degree_from_mat(mat_file)
        true_ls.append(yaw)
        print(yaw)
        img = utils.crop_image(mat_file, jpg_path)
        img = img_to_array(img.resize((img_size, img_size)))

        # 0-1に変換
        img_nad = img/255
        # 4次元配列に
        img_nad = np.expand_dims(img_nad, axis=0)
        # 画像のロード
        result = model.predict(img_nad)[0]
        result_ls.append(result)

    # listをnumpyに変換
    result_np = np.array(result_ls)
    true_np = np.array(true_ls)

    save_path = './model/output/' + args.model_path + '/'
    # save numpy as csv
    np.savetxt(save_path + 'result_np_val.csv', result_np, delimiter=',')
    np.savetxt(save_path + 'true_np_val.csv', true_np, delimiter=',')

    # calculate x
    tmp_np = np.dot(result_np.T, result_np)
    # reverse
    tmp_np = np.linalg.inv(tmp_np)
    tmp_np = np.dot(tmp_np, result_np.T)
    x_np = np.dot(tmp_np, true_np)
    np.savetxt(save_path + 'x_np.csv', tmp_np,delimiter=',')


