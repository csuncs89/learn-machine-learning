from __future__ import print_function

import abc
import hashlib
import json
import os
import sys

import cv2
import keras
import numpy as np
import sklearn.utils
from keras import callbacks


def normal_show(win_name, img):
    """Show image in a normal window

    @param win_name: Window name
    @param img: The image
    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)


def normal_show_wait_esc(win_name, img):
    """Show image in a normal window and wait for key, ESC to exit

    @param win_name: Window name
    @param img: The image
    """
    normal_show(win_name, img)
    wait_esc()


def wait_esc():
    """Wait for key, ESC to exit
    """
    key = cv2.waitKey()
    if key == 27:
        sys.exit(0)
    return key


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

        # str (Base directory to save the running results)
        self.dir_base = None

    def _load_dir(self, path):
        print('Scanning subdirectory', path)
        imgs = []
        labels = []
        names = []
        for name_label in sorted(os.listdir(path)):
            if not name_label.isdigit():
                continue
            label = int(name_label)
            path1 = os.path.join(path, name_label)

            for name_img in sorted(os.listdir(path1)):
                path_img = os.path.join(path1, name_img)
                img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.w_img, self.h_img))
                if self.invert_img:
                    img = 255 - img
                imgs.append(img)
                labels.append(label)

                names.append(name_label + name_img)

        m = hashlib.md5()
        print('Length of names: {0}'.format(len(names)))
        m.update(''.join(names).encode('utf-8'))
        val_md5 = m.hexdigest()
        print(val_md5)

        return np.array(imgs), np.array(labels), val_md5

    def _load_data(self, dir_dataset):
        dir_train = os.path.join(dir_dataset, 'train')
        x_data, y_data, md5_data = self._load_dir(dir_train)
        assert (md5_data == 'c2350a4dff799934bcfa5be7193dd91e')
        x_data, y_data = sklearn.utils.shuffle(x_data, y_data, random_state=0)
        num_validation = int(round(len(x_data) * self.percent_validation))
        x_train, y_train = x_data[num_validation:], y_data[num_validation:]
        x_validation, y_validation = \
            x_data[:num_validation], y_data[:num_validation]

        dir_test = os.path.join(dir_dataset, 'test')
        x_test, y_test, md5_test = self._load_dir(dir_test)
        assert (md5_test == '4a19afda4bd091b6d746166b57859267')

        x_train = x_train.reshape(x_train.shape[0], self.h_img, self.w_img, 1)
        x_validation = x_validation.reshape(x_validation.shape[0], self.h_img,
                                            self.w_img, 1)
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
        y_validation = keras.utils.to_categorical(y_validation,
                                                  self.num_classes)
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
        path_weights = os.path.join(self.dir_base, self.version_recognizer,
                                    'weights.hdfs')
        self._path_weights = path_weights

    def _load_weights(self):
        if self._path_weights is None:
            self._set_path_weights()
        if os.path.exists(self._path_weights):
            self._model.load_weights(self._path_weights)
            print('Weights are restored from', self._path_weights)

    def _train(self, epochs):
        self._load_weights()
        callback_checkpoint = callbacks.ModelCheckpoint(
            filepath=self._path_weights,
            verbose=1,
            save_best_only=True,
            save_weights_only=True)

        dir_log_tensorboard = os.path.join(self.dir_base,
                                           self.version_recognizer,
                                           'log_tensorboard')
        if not os.path.exists(dir_log_tensorboard):
            os.makedirs(dir_log_tensorboard)
        callback_tensorboard = callbacks.TensorBoard(
            log_dir=dir_log_tensorboard)

        self._model.fit(self.x_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(self.x_validation, self.y_validation),
                        callbacks=[callback_checkpoint, callback_tensorboard])

    def _evaluate_test_set(self, load_weights=True):
        if load_weights:
            self._load_weights()
        score = self._model.evaluate(self.x_test, self.y_test, verbose=0)
        path_test_results = os.path.join(self.dir_base, self.version_recognizer,
                                         'test_results.json')
        json.dump({'test_loss': score[0], 'test_accuracy': score[1]},
                  open(path_test_results, 'w'), indent=4, sort_keys=True)
        print('Test loss:', score[0])
        print('Test accuracy', score[1])

    def run(self, dir_dataset, epochs):
        dir_result = os.path.join(self.dir_base, self.version_recognizer)
        if os.path.exists(dir_result):
            print('Result directory {0} exists, skip run recognizer {1}'
                  .format(dir_result, self.version_recognizer))
            return

        print('Run recognizer {0} ...'.format(self.version_recognizer))
        self._load_data(dir_dataset)
        self._create_model()
        self._train(epochs=epochs)
        self._evaluate_test_set()

    def print_model_arch(self):
        print('Model definition:')
        self._create_model()
        print(self._model.summary())
        print(self._model.to_yaml())

    def validate_in_detail(self, dir_dataset):
        self._load_data(dir_dataset)
        self._create_model()
        self._load_weights()

        print('Evaluate on the validation set')
        val_loss, val_acc = self._model.evaluate(self.x_validation,
                                                 self.y_validation,
                                                 verbose=1)
        print('Validation accuracy: {0}'.format(val_acc))
        print('Validation error rate: {0}'.format(1 - val_acc))
        print('Number of incorrect predictions: {0}'
              .format(len(self.x_validation) * (1 - val_acc)))

        y_validation_pred = np.argmax(self._model.predict(self.x_validation),
                                      axis=1)
        y_validation = np.argmax(self.y_validation, axis=1)

        print(y_validation_pred.shape)
        print(y_validation.shape)

        for i, (y_gt, y_pred) in enumerate(zip(y_validation,
                                               y_validation_pred)):
            if y_gt != y_pred:
                x = self.x_validation[i]
                img = np.asarray(255 - (x * 255.0), np.uint8)
                print('predicated label {0} != gt label {1}'
                      .format(y_pred, y_gt))

                dir_train = os.path.join(dir_dataset, 'train', str(y_pred))
                path_train = os.path.join(dir_train, os.listdir(dir_train)[1])
                print(path_train)
                normal_show('img_pred', cv2.imread(path_train,
                                                   cv2.IMREAD_GRAYSCALE))
                normal_show_wait_esc('img', img)

    def train_full_train_data(self, dir_dataset):
        dir_result = os.path.join(self.dir_base, self.version_recognizer)

        print('Run recognizer {0} ...'.format(self.version_recognizer))
        self._load_data(dir_dataset)
        self._create_model()
        self.x_train = np.concatenate([self.x_train, self.x_validation])
        self.y_train = np.concatenate([self.y_train, self.y_validation])
        self._load_weights()
        self._model.fit(self.x_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=2,
                        verbose=1)
        self._evaluate_test_set(load_weights=False)
