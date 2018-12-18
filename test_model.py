import glob
from data_maker import utils
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from model import googlenet2

if __name__  == '__main__':
    weight_path = './model/output/my_model.hdf5'
    dataset_path = '../dataset/AFLW2000'

    img_size = 150
    # モデルをコンパイル
    model = googlenet2.create_googlenet(
        img_size = img_size)
    model.load_weights(weight_path)

    positive_num = 0
    test_num = 0
    degree_th = 10

    mat_files = glob.glob(dataset_path +'/*.mat')
    jpg_images = glob.glob(dataset_path +'/*.jpg')

    np_degree = np.arange(-90, 90, 10)
    for mat_file, jpg_image in zip(mat_files, jpg_images):
        pitch, yaw, roll = utils.get_degree_from_mat(mat_file)
        
        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            test_num += 1
            print(jpg_image)
            img = img_to_array(load_img(jpg_image, target_size=(img_size, img_size)))
            #0-1に変換
            img_nad = img_to_array(img)/255
            #4次元配列に
            img_nad = img_nad[None, ...]
            results = model.predict(img_nad)
            print(results)
            print('result : ' + str(np.dot(np_degree, results[0])))
            print('correct : ' + str(yaw))
            
            if test_num == 3:
                break