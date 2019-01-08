import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from IPython.display import SVG
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, Input, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model

from PIL import Image

from model import googlenet2
import utils_for_keras


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--model_path', dest='model_path',
                        help='String appended to output model.', default='1', type=str)
    parser.add_argument('--output', dest='output_folder', default='1',
                        help='Folder name.', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.001, type=float)
    parser.add_argument('--momentum', dest='momentum', help='',
                        default=0.9, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    weight_path = './model/output/' + args.model_path + '/my_model.hdf5'

    test_datagen = ImageDataGenerator()
    dataset_path = './dataset/divided/'
    img_size = 150
    test_generator = test_datagen.flow_from_directory(
        dataset_path + 'test',
        target_size=(img_size, img_size),
        batch_size=args.batch_size,
        class_mode='categorical')

    # モデルをコンパイル
    model = VGG16(weights='imagenet',
                  include_top=False,
                  input_tensor=Input(shape=(img_size, img_size, 3))
                  )

    y = Flatten()(model.output)

    y = Dense(800, activation='relu')(y)
    y = Dense(800, activation='relu')(y)
    y = Dense(18, activation='softmax')(y)

    model = Model(inputs=model.input, outputs=y)
    model.load_weights(weight_path)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.SGD(
                      lr=args.lr, momentum=args.momentum),
                  metrics=["accuracy"])

    loss, acc = model.evaluate_generator(test_generator, steps=10)
    print('Test loss: %.4f, Test acc: %.4f' % (loss, acc))
