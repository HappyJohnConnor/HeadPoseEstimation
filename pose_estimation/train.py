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

from PIL import Image

from data import Pose_300W_LP
from model import googlenet2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=100, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.001, type=float)
    parser.add_argument('--momentum', dest='momentum', help='',
                        default=0.9, type=float)
    parser.add_argument('--model_name', dest='model_name',
                        help='String appended to output model.', default='my_model', type=str)
    parser.add_argument('--outcome_name', dest='outcome_name', help='String appended to output outcome.',
                        default='my_outcome', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    image_list, label_list = Pose_300W_LP()
    print(len(label_list))
    print(len(image_list))

    # kerasに渡すためにnumpy配列に変換。
    image_list = np.array(image_list)

    # ラベルの配列を1と0からなるラベル配列に変更
    # 0 -> [1,0], 1 -> [0,1] という感じ。
    Y = to_categorical(label_list)
    print(Y.shape)

    # モデルをコンパイル
    model = googlenet2.create_googlenet('./model/googlenet_weights.h5')
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.SGD(
                      lr=args.lr, momentum=args.momentum),
                  metrics=["accuracy"])

    # 学習を実行。10%はテストに使用。
    print(image_list[0].shape)
    history = model.fit(image_list,
                        Y,
                        epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        validation_split=0.1
                        )

    # モデルの保存
    model_save_path = './model/output'
    json_string = model.to_json()
    open(os.path.join(model_save_path, args.model_name + '.json'),
         'w').write(json_string)
    yaml_string = model.to_yaml()
    open(os.path.join(model_save_path, args.model_name + '.yaml'),
         'w').write(yaml_string)
    print('save weights')
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
