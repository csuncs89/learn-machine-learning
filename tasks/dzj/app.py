"""

"""
from __future__ import print_function

import abc
import argparse
import json
import os
import sys

import cv2
import keras
import numpy as np
import sklearn.utils
import tensorflow as tf
from keras import callbacks
from keras import layers
from keras import models


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


class DzjRecognizer(abc.ABC):

    def __init__(self):
        self.x_train, self.y_train = None, None
        self.x_validation, self.y_validation = None, None
        self.x_test, self.y_test = None, None
        self._model = None
        self._path_weights = None
        self._configure()

    @abc.abstractmethod
    def _configure(self):
        """Configure the recognizer
        """
        # int (Width of the input image)
        self.w_img = None
        # int (Height of the input image)
        self.h_img = None

        # bool (Whether to invert the input image)
        self.invert_img = None

        # int (Number of classes)
        self.num_classes = None

        # int (Batch size)
        self.batch_size = None

        # str (Version of the recognizer)
        self.version_recognizer = None

        # bool (Whether to enable local debugging)
        self.local_debug = None

        # bool (Whether to normalize input image so that mean is 0)
        self.normalize_img_mean0 = None

        # float (Percent of validation data)
        self.percent_validation = None

    def _load_dir(self, path):
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
                img = cv2.resize(img, (self.w_img, self.h_img))
                if self.invert_img:
                    img = 255 - img
                imgs.append(img)
                labels.append(label)
        return np.array(imgs), np.array(labels)

    def _load_data(self, dir_dataset):
        dir_train = os.path.join(dir_dataset, 'train')
        x_data, y_data = self._load_dir(dir_train)
        x_data, y_data = sklearn.utils.shuffle(x_data, y_data, random_state=0)
        num_validation = int(round(len(x_data) * self.percent_validation))
        x_train, y_train = x_data[num_validation:], y_data[num_validation:]
        x_validation, y_validation = x_data[:num_validation], y_data[:num_validation]

        dir_test = os.path.join(dir_dataset, 'test')
        x_test, y_test = self._load_dir(dir_test)

        x_train = x_train.reshape(x_train.shape[0], self.h_img, self.w_img, 1)
        x_validation = x_validation.reshape(x_validation.shape[0], self.h_img, self.w_img, 1)
        x_test = x_test.reshape(x_test.shape[0], self.h_img, self.w_img, 1)

        if self.local_debug:
            x_train = x_train[:1000]
            y_train = y_train[:1000]
            x_test = x_test[:100]
            y_test = y_test[:100]

        if self.normalize_img_mean0:
            x_train = (x_train.astype('float32') - 128) / 100.0
            x_validation = (x_validation.astype('float32') - 128) / 100.0
            x_test = (x_test.astype('float32') - 128) / 100.0
        else:
            x_train = x_train.astype('float32') / 255
            x_validation = x_validation.astype('float32') / 255
            x_test = x_test.astype('float32') / 255
        print('x_train.shape:', x_train.shape)
        print('x_validation.shape:', x_validation.shape)
        print('x_test.shape:', x_test.shape)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_validation = keras.utils.to_categorical(y_validation, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.x_train, self.y_train = x_train, y_train
        self.x_validation, self.y_validation = x_validation, y_validation
        self.x_test, self.y_test = x_test, y_test

    @abc.abstractmethod
    def _create_model(self):
        """Create the model
        Set the self._model here
        """
        self._model = None

    def _set_path_weights(self):
        path_weights = os.path.join(self.version_recognizer, 'weights.hdfs')
        self._path_weights = path_weights

    def _load_weights(self):
        if self._path_weights is None:
            self._set_path_weights()
        if os.path.exists(self._path_weights):
            self._model.load_weights(self._path_weights)
            print('Weights are restored from', self._path_weights)

    def _train(self, epochs):
        self._load_weights()
        callback_checkpoint = callbacks.ModelCheckpoint(filepath=self._path_weights,
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=True)

        dir_log_tensorboard = os.path.join(self.version_recognizer, 'log_tensorboard')
        if not os.path.exists(dir_log_tensorboard):
            os.makedirs(dir_log_tensorboard)
        callback_tensorboard = callbacks.TensorBoard(log_dir=dir_log_tensorboard)

        self._model.fit(self.x_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(self.x_validation, self.y_validation),
                        callbacks=[callback_checkpoint, callback_tensorboard])

    def _evaluate(self):
        self._load_weights()
        score = self._model.evaluate(self.x_test, self.y_test, verbose=0)
        path_test_results = os.path.join(self.version_recognizer, 'test_results.json')
        json.dump({'test_loss': score[0], 'test_accuracy': score[1]},
                  open(path_test_results, 'w'), indent=4, sort_keys=True)
        print('Test loss:', score[0])
        print('Test accuracy', score[1])

    def run(self, dir_dataset, epochs):
        print('Run recognizer {0} ...'.format(self.version_recognizer))
        self._load_data(dir_dataset)
        self._create_model()
        self._train(epochs=epochs)
        self._evaluate()


class DzjRecognizerV1(DzjRecognizer):

    def _configure(self):
        self.w_img = 28
        self.h_img = 28
        self.invert_img = True
        self.num_classes = 200
        self.batch_size = 64
        self.version_recognizer = 'v1'
        self.local_debug = False
        self.normalize_img_mean0 = False
        self.percent_validation = 0.1

    def _create_model(self):
        """
        3x3x32 3x3x64 2x2 128 200
        """
        input_shape = (self.h_img, self.w_img, 1)
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
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV2(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV2, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v2'

    def _create_model(self):
        """
        3x3x32 3x3x64 2x2 3x3x64 2x2 128 200
        """
        input_shape = (self.h_img, self.w_img, 1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV3(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV3, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v3'

    def _create_model(self):
        """
        3x3x32 3x3x64 2x2 3x3x64 2x2 500 200
        """
        input_shape = (self.h_img, self.w_img, 1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV4(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV4, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v4'

    def _create_model(self):
        """
        3x3x32 3x3x64 2x2 3x3x64 2x2 500 500 200
        """
        input_shape = (self.h_img, self.w_img, 1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV5(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV5, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v5'

    def _create_model(self):
        """
        3x3x32 3x3x64 2x2 3x3x64 2x2 128 128 200
        """
        input_shape = (self.h_img, self.w_img, 1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV6(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV6, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v6'

    def _create_model(self):
        """
        3x3x32 3x3x64 2x2 3x3x64 2x2 500 500 200
        """
        input_shape = (self.h_img, self.w_img, 1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1000, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV7(DzjRecognizerV4):

    def _configure(self):
        super(DzjRecognizerV7, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v7'

        self.normalize_img_mean0 = True


class DzjRecognizerV8(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV8, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v8'

        self.normalize_img_mean0 = True

    def _create_model(self):
        """
        3x3x32 3x3x64 3x3x64 2x2 3x3x64 2x2 500 500 200
        """
        input_shape = (self.h_img, self.w_img, 1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV9(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV9, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v9'

        self.normalize_img_mean0 = True

    def _create_model(self):
        """
        3x3x32 3x3x64 3x3x64 2x2 3x3x64 2x2 500 500 500 200
        """
        input_shape = (self.h_img, self.w_img, 1)
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(500, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


class DzjRecognizerV10(DzjRecognizerV1):
    def _configure(self):
        super(DzjRecognizerV10, self)._configure()

        # Version of the recognizer
        self.version_recognizer = 'v10'

        self.normalize_img_mean0 = True

    def _create_model(self):
        """
        3x3x32 3x3x64 2x2 3x3x64 2x2 500 500 200
        """
        input_shape = (self.h_img, self.w_img, 1)

        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), input_shape=input_shape, use_bias=False))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(64, (3, 3), use_bias=False))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())

        model.add(layers.Dense(500, use_bias=False))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(500, use_bias=False))
        model.add(layers.BatchNormalization(axis=-1))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        self._model = model


def main():
    args = parse_args()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    for id_recognizer in range(10, 11):
        name_class = 'DzjRecognizerV' + str(id_recognizer)
        recognizer = globals()[name_class]()
        recognizer.run(args.dir_dataset, epochs=100)


if __name__ == '__main__':
    main()
