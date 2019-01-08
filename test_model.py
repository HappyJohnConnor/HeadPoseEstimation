import argparse
import glob
from data_maker import utils
import numpy as np

from matplotlib import pylab as plt
from keras.preprocessing.image import img_to_array
import utils_for_keras


def parse_args():
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--model_path', dest='model_path',
                        help='String appended to output model.', default='1', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_path = '../dataset/AFLW2000'

    img_size = 150
    weight_path = './model/output/' + args.model_path + '/my_model.hdf5'
    model = utils_for_keras.get_model(weight_path=weight_path)

    positive_num = 0
    test_count = 0
    correct_count = 0
    degree_th = 10

    diff_ls = []

    np_degree = np.array([-10, -20, -30, -40, -50, -60, -
                          70, -80, -90, 0, 10, 20, 30, 40, 50, 60, 70, 80])

    np_degree_5 = np.array([-5, -15, -25, -35, -45, -55, -
                            65, -75, -85, 5, 15, 25, 35, 45, 55, 65, 75, 85])

    jpg_images = glob.glob(dataset_path + '/*.jpg')

    # load csv file to numpy
    result_np = np.loadtxt('./model/output/' +
                           args.model_path + '/result_np.csv', delimiter=",")
    true_np = np.loadtxt('./model/output/' +
                         args.model_path + '/true_np.csv', delimiter=",")
    # calculate x
    process_np = np.dot(result_np.T, result_np)
    # reverse
    process_np = np.linalg.inv(process_np)
    process_np = np.dot(process_np, result_np.T)
    optical_coef = np.dot(process_np, true_np)

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
            if np_degree[max_idx] == int(yaw - yaw % 10):
                correct_count += 1

            # diff_ls.append(abs(yaw - np.dot(optical_coef, results[0])))
            diff_ls.append(abs(yaw - np.dot(np_degree_5, results[0])))

            print('----------------------')
            print('test count : %d' % test_count)
            print('correct   degree : %d' % yaw)
            print('predicted degree : %d' % np_degree[max_idx])
            print('average   degree : %d' % (np.dot(optical_coef, results[0])))
            # print('average   degree : %d' % (np.dot(np_degree_5, results[0])))
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
    plt.hist(diff_np, bins=36)
    plt.xlim(0, 180)
    plt.xlabel("degree")
    plt.ylabel('number')
    plt.savefig("hoge2.png")
