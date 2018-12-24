import glob
from data_maker import utils
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Activation, Dense, Dropout, Input, Flatten
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from data_maker import utils

#from model import googlenet2

if __name__ == '__main__':
    weight_path = './model/output/my_model.hdf5'
    dataset_path = '../dataset/AFLW2000'

    img_size = 150
    """
    # モデルをコンパイル
    model = googlenet2.create_googlenet(
        img_size = img_size)
    model.load_weights(weight_path)
    """
    # モデルをコンパイル
    model = VGG16(
        include_top=False,
        input_tensor=Input(shape=(img_size, img_size, 3))
    )

    y = Flatten()(model.output)

    y = Dense(800, activation='relu')(y)
    y = Dense(800, activation='relu')(y)
    y = Dense(18, activation='softmax')(y)

    model = Model(inputs=model.input, outputs=y)
    model.load_weights(weight_path)

    positive_num = 0
    test_count = 0
    degree_th = 10

    mat_files = glob.glob(dataset_path + '/*.mat')
    jpg_images = glob.glob(dataset_path + '/*.jpg')

    np_degree = np.array([-10, -20, -30, -40, -50, -60, -
                          70, -80, -90, 0, 10, 20, 30, 40, 50, 60, 70, 80])
    print(np_degree)
    for jpg_image in jpg_images:
        mat_file = utils.get_matpath(jpg_image)
        pitch, yaw, roll = utils.get_degree_from_mat(mat_file)

        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            test_count += 1
            img = utils.crop_image(mat_file, jpg_image)
            img = img_to_array(img.resize((img_size, img_size)))
            # 0-1に変換
            img_nad = img/255
            # 4次元配列に
            img_nad = np.expand_dims(img_nad, axis=0)
            results = model.predict(img_nad)
            # 最大値を返す
            max_idx = results[0].argmax()
            print(type(results))
            print('test count : %d' % test_count)
            #print('degree : %d' % np_degree(max_idx))
            print('result : ' + str(np.dot(np_degree, results[0])))
            print('correct : ' + str(yaw))

            if test_count == 10:
                break
