from keras.applications.vgg16 import VGG16
from keras.layers import Activation, Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
import numpy as np


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


# モデルを返す
def get_model(weight_path, img_size=150, output_num = 18):
    # モデルのロード
    model = VGG16(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=(img_size, img_size, 3))
    )

    # 最後の畳み込み層より前の層の再学習を防止
    for layer in model.layers:
        layer.trainable = False

    y = Flatten()(model.output)

    y = Dense(800, activation='relu')(y)
    y = Dense(800, activation='relu')(y)
    y = Dense(output_num, activation='softmax')(y)

    model = Model(model.input, y)

    if weight_path:
        model.load_weights(weight_path)

    return model
