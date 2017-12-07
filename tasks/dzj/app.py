"""

"""
from __future__ import print_function

import argparse
import os
import sys

import keras
import tensorflow as tf
from keras import layers
from keras import models

from tasks.dzj import dzj_model


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


class DzjRecognizerV1(dzj_model.DzjRecognizer):

    def _configure(self):
        self.w_img = 28
        self.h_img = 28
        self.invert_img = True
        self.num_classes = 200
        self.batch_size = 64
        self.version_recognizer = self.__class__.__name__[len('DzjRecognizer'):]
        self.local_debug = False
        self.normalize_img_mean0 = False
        self.percent_validation = 0.1
        self.dir_base = os.path.join(os.getenv('HOME'), 'dzj_results',
                                     os.path.basename(__file__)[:-3])

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
        self.normalize_img_mean0 = True


class DzjRecognizerV8(DzjRecognizerV1):

    def _configure(self):
        super(DzjRecognizerV8, self)._configure()
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

    for id_recognizer in range(1, 11):
        name_class = 'DzjRecognizerV' + str(id_recognizer)
        recognizer = globals()[name_class]()
        recognizer.run(args.dir_dataset, epochs=100)


if __name__ == '__main__':
    main()
