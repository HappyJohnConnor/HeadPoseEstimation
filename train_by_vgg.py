import argparse
import matplotlib.pyplot as plt
import os
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import utils_for_keras


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=200, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=128, type=int)
    parser.add_argument('--direction', dest='direction', default='yaw', type=str)
    parser.add_argument('--output_num', dest='output_num', default=18, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.001, type=float)
    parser.add_argument('--momentum', dest='momentum', help='',
                        default=0.9, type=float)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='Folder name.', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        preprocessing_function=utils_for_keras.get_random_eraser(v_l=0, v_h=1, pixel_level=False))

    val_datagen = ImageDataGenerator(rescale=1./255)

    dataset_path = './dataset/'+ args.direction + '/'
    img_size = 150
    train_generator = train_datagen.flow_from_directory(
        dataset_path + 'train',
        target_size=(img_size, img_size),
        batch_size=args.batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        dataset_path + 'valid',
        target_size=(img_size, img_size),
        batch_size=args.batch_size,
        class_mode='categorical')
    
    img_size = 150
    model = utils_for_keras.get_model(
        img_size = img_size,
        output_num = args.output_num
    )

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.Adam(lr=args.lr),
                  metrics=["accuracy"])

    es_cb = EarlyStopping(monitor='val_loss', verbose=0, mode='auto')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=args.num_epochs,
        validation_data=validation_generator,
        validation_steps=200,
        #callbacks=[es_cb]
    )

    # モデルの保存
    model_save_path = './model/output/' + args.direction + '/' + args.output_folder + '/'
    os.makedirs(model_save_path, exist_ok=True)
    model.save_weights(os.path.join(
        model_save_path, 'my_model.hdf5'))

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
    plt.savefig(model_save_path + "acc.png")
    plt.close()

    # plot loss
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.legend()
    plt.savefig(model_save_path + "loss.png")
    plt.close()
