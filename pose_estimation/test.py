import glob
import data_utils
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

from model import googlenet2

if __name__  == '__main__':
    weight_path = './model/output/my_model.hdf5'
    dataset_path = './dataset/test/AFLW2000'

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
        pitch, yaw, roll = data_utils.get_degree_from_mat(mat_file)
        
        if abs(pitch) <= degree_th and abs(roll) <= degree_th:
            test_num += 1
            
            img = image.load_img(jpg_image, target_size=(img_size, img_size))
            x = image.img_to_array(img)
            x = x.reshape((-1, x.shape[0], x.shape[1], x.shape[2]))
            print(x)            
            results = model.predict(x/255.)
            print(results)
            print('result : ' + str(np.dot(np_degree, results[0])))
            print('correct : ' + str(yaw))
            
            if test_num == 3:
                break