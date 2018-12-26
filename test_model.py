import argparse
import glob
from data_maker import utils
import numpy as np

from matplotlib import pylab as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Activation, Dense, Dropout, Input, Flatten
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from data_maker import utils

#from model import googlenet2
def parse_args():
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--model_path', dest='model_path',
                        help='String appended to output model.', default='1', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    weight_path = './model/output/' + args.model_path +  '/my_model.hdf5'
    
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

    model = Model(model.input, y)
    model.load_weights(weight_path)

    positive_num = 0
    test_count = 0
    correct_count  = 0
    degree_th = 10

    diff_ls = []

    np_degree = np.array([-10, -20, -30, -40, -50, -60, -
                          70, -80, -90, 0, 10, 20, 30, 40, 50, 60, 70, 80])

    jpg_images = glob.glob(dataset_path + '/*.jpg')

    for jpg_image in jpg_images:
        mat_file = utils.get_matpath(jpg_image)
        pitch, yaw, roll = utils.get_degree_from_mat(mat_file)

        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            test_count += 1
            img = utils.crop_image(mat_file, jpg_image)
            img.show()
            img = img_to_array(img.resize((img_size, img_size)))
            
            # 0-1に変換
            img_nad = img/255
            # 4次元配列に
            img_nad = np.expand_dims(img_nad, axis=0)
            results = model.predict(img_nad)
            # 最大値を返す
            max_idx = results[0].argmax()
            if np_degree[max_idx] == int(yaw- yaw % 10):
                correct_count += 1

            diff_ls.append(abs(yaw - np.dot(np_degree, results[0])))
            
            print('----------------------')
            print('test count : %d' % test_count)
            print('correct   degree : %d' % yaw)
            print('predicted degree : %d' % np_degree[max_idx])
            print('average   degree : %d' % (np.dot(np_degree, results[0])))
            print('correct count : %d' % correct_count)
            
            """
            if test_count == 3:
                break
            """
            
     
    diff_np = np.array(diff_ls)
    print('standard deviation : %.2f' % np.std(diff_np))
    print('mean               : %.2f' % np.mean(diff_np))

    hist, bins = np.histogram(diff_np, bins=18)

    # -90〜90までplot
    plt.hist(diff_np, bins = 36)
    plt.xlim(0, 180)
    plt.xlabel("degree")
    plt.ylabel('number')
    plt.savefig("hoge.png")

     

