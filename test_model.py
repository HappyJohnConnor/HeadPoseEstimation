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
    parser.add_argument('--direction', dest='direction',
                        default='yaw', type=str)
    parser.add_argument('--output_num', dest='output_num',
                        default=18, type=int)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='Folder name.', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_path = '../dataset/AFLW2000'

    img_size = 150
    weight_path = './model/output/' + args.direction + \
        '/' + args.output_folder + '/my_model.hdf5'
    model = utils_for_keras.get_model(
        weight_path=weight_path, output_num=args.output_num)

    positive_num = 0
    test_count = 0
    correct_count = 0
    degree_th = 90
    fail_count = 0
    txt_path = 'result.txt'
    diff_ls = []

    np_degree = np.array([-10, -20, -30, -40, -50, -60, -
                          70, -80, -90, 0, 10, 20, 30, 40, 50, 60, 70, 80])

    np_degree_5 = np.array([-5, -15, -25, -35, -45, -55, -
                            65, -75, -85, 5, 15, 25, 35, 45, 55, 65, 75, 85])

    jpg_images = glob.glob(dataset_path + '/*.jpg')

    model_path = './model/output/' + args.direction + '/' + args.output_folder + '/'
    # load csv file to numpy
    result_np = np.loadtxt(model_path + 'result_np.csv', delimiter=",")
    true_np = np.loadtxt(model_path + 'true_np.csv', delimiter=",")
    # calculate x
    process_np = np.dot(result_np.T, result_np)
    # reverse
    process_np = np.linalg.inv(process_np)
    process_np = np.dot(process_np, result_np.T)
    optical_coef = np.dot(process_np, true_np)

    designed_deg = None
    with open(txt_path, 'w') as f:
        for jpg_image in jpg_images:
            mat_file = utils.get_matpath(jpg_image)
            pitch_deg, yaw_deg, roll_deg = utils.get_degree_from_mat(mat_file)
            if args.direction == 'pitch':
                designed_deg = pitch_deg
            elif args.direction == 'yaw':
                designed_deg = yaw_deg
            elif args.direction == 'roll':
                designed_deg = roll_deg

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
            if np_degree[max_idx] == int(designed_deg - designed_deg % 10):
                correct_count += 1

            diff_ls.append(
                abs(designed_deg - np.dot(optical_coef, results[0])))

            print('----------------------')
            print('test count : %d' % test_count)
            print('correct   degree : %d' % designed_deg)
            print('predicted degree : %d' % np_degree[max_idx])
            print('average   degree : %d' % (np.dot(optical_coef, results[0])))
            print('correct count : %d' % correct_count)

            if abs(designed_deg - np.dot(optical_coef, results[0])) >= 70 and fail_count <= 3:
                fail_count += 1
                img_name = utils.get_img_name(jpg_image)
                s_correct = 'correct   degree : {:.2f}'.format(designed_deg)
                s_failure = 'predicted degree : {:.2f}'.format(
                    np.dot(optical_coef, results[0]))
                sentence = '{}\n{}\n{}\n'.format(
                    img_name, s_correct, s_failure)

                f.write(sentence)
            """
            if test_count == 3:
                break
            """

    diff_np = np.array(diff_ls)
    print('standard deviation : %.2f' % np.std(diff_np))
    print('mean               : %.2f' % np.mean(diff_np))

    # -90〜90までplot
    plt.hist(diff_np, bins=181)
    plt.xlim(0, 180)
    plt.xlabel("degree")
    plt.ylabel('number')
    plt.savefig(model_path + "test_hist.png")
