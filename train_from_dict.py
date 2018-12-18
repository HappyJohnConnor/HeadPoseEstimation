import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from IPython.display import SVG

from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from data import Pose_300W_LP
from model import googlenet2
import util

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default = 100, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default = 128, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.001, type=float)
    parser.add_argument('--momentum', dest='momentum', help='',
                        default = 0.9, type=float)
    parser.add_argument('--model_name', dest='model_name',
                        help='String appended to output model.', default='my_model', type=str)
    parser.add_argument('--outcome_name', dest='outcome_name', help='String appended to output outcome.',
                        default = 'my_outcome', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=util.get_random_eraser(v_l=0, v_h=1, pixel_level=False))

    val_datagen = ImageDataGenerator(rescale=1./255)

    dataset_path = '../300w_lp/dataset/divided/'
    img_size = 150
    train_generator = train_datagen.flow_from_directory(
            dataset_path + 'train',
            target_size=(img_size, img_size),
            batch_size= args.batch_size,
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            dataset_path + 'valid',
            target_size=(img_size, img_size),
            batch_size= args.batch_size,
            class_mode='categorical')

    # モデルをコンパイル
    model = googlenet2.create_googlenet(
        img_size = img_size,
        weights_path = './model/googlenet_weights.h5')
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.SGD(
                      lr=args.lr, momentum=args.momentum),
                  metrics=["accuracy"])

    # 学習を実行。20%はテストに使用。
    history = model.fit_generator(
        train_generator, 
        steps_per_epoch=200,
        epochs=args.num_epochs,
        validation_data = validation_generator,
        validation_steps=200
    )

    # モデルの保存
    model_save_path = './model/output'
    model.save_weights(os.path.join(
        model_save_path, args.model_name + '.hdf5'))

    # Plot accuracy &amp; loss

    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    # plot accuracy
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and Validation accuracy")
    plt.legend()
    plt.savefig("acc.png")
    plt.close()

    # plot loss
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()
