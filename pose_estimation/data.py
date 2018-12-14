from dataset import util
import glob
import numpy as np
import os


def Pose_300W_LP():
    """
    画像リストとラベルリストを返す
    """
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

    return image_list, label_list
