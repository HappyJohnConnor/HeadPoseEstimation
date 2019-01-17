import argparse
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from pathlib import Path

from data_maker import utils
import utils_for_keras


def get_matpath(img_path):
    base_path = './dataset/mat'
    _, file_name = os.path.split(img_path)
    name_ls = file_name.split('.')
    new_name = name_ls[0]+'.mat'
    mat_path = base_path + '/' + new_name

    return mat_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--output_folder', dest='output_folder',
                        help='String appended to output model.', default='1', type=str)
    parser.add_argument('--direction', dest='direction', default='yaw', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_save_path = './model/output/' + args.direction + '/' + args.output_folder + '/'
    weight_path = model_save_path + 'my_model.hdf5'
    img_size = 150
    model = utils_for_keras.get_model(weight_path=weight_path)

    # Pathオブジェクトを生成
    base_path = Path("./dataset/" + args.direction + "/train/")
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

    # save numpy as csv
    np.savetxt(model_save_path + 'result_np.csv', result_np, delimiter=',')
    np.savetxt(model_save_path + 'true_np.csv', true_np, delimiter=',')

    # calculate x
    tmp_np = np.dot(result_np.T, result_np)
    # reverse
    tmp_np = np.linalg.inv(tmp_np)
    tmp_np = np.dot(tmp_np, result_np.T)
    x_np = np.dot(tmp_np, true_np)
    np.savetxt(model_save_path + 'x_np.csv', tmp_np, delimiter=',')
