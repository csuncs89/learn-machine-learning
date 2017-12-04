"""

"""
from __future__ import print_function

import argparse
import os
import sys

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import callbacks
from keras import layers
from keras import models

LOCAL_DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Character recognition'
    )
    parser.add_argument(
        'dir_dataset',
        help='Directory to the dataset',
    )

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    return args


def load_dir(path):
    print('Scanning subdirectory', path)
    imgs = []
    labels = []
    for name_label in os.listdir(path):
        if not name_label.isdigit():
            continue
        label = int(name_label)
        path1 = os.path.join(path, name_label)

        for name_img in os.listdir(path1):
            path_img = os.path.join(path1, name_img)
            img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = 255 - img
            imgs.append(img)
            labels.append(label)
    return np.array(imgs), np.array(labels)


def main():
    args = parse_args()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    batch_size = 64
    num_classes = 200
    epochs = 100

    img_h, img_w = 28, 28

    dir_train = os.path.join(args.dir_dataset, 'train')
    x_train, y_train = load_dir(dir_train)

    dir_test = os.path.join(args.dir_dataset, 'test')
    x_test, y_test = load_dir(dir_test)

    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    input_shape = (img_h, img_w, 1)

    if LOCAL_DEBUG:
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_test = x_test[:100]
        y_test = y_test[:100]

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print('x_train.shape:', x_train.shape)
    print('x_test.shape:', x_test.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    dir_version = 'v1'
    path_weights = os.path.join(dir_version, 'weights.hdfs')
    if os.path.exists(path_weights):
        model.load_weights(path_weights)
        print('Weights are restored from', path_weights)

    callback_checkpoint = callbacks.ModelCheckpoint(filepath=path_weights,
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True)

    dir_log_tensorboard = os.path.join(dir_version, 'log_tensorboard')
    if not os.path.exists(dir_log_tensorboard):
        os.makedirs(dir_log_tensorboard)
    callback_tensorboard = callbacks.TensorBoard(log_dir=dir_log_tensorboard)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[callback_checkpoint, callback_tensorboard])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy', score[1])


if __name__ == '__main__':
    main()
