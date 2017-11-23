"""

"""
import pickle

import numpy as np

from sklearn import datasets
from sklearn import random_projection
from sklearn import svm


def split_dataset(X, ys, num_training):
    X_training = X[:num_training]
    ys_training = ys[:num_training]
    X_test = X[num_training:]
    ys_test = ys[num_training:]

    return (X_training, ys_training), (X_test, ys_test)


def learn_and_predict():
    digits = datasets.load_digits()

    print(digits.data)

    classifier = svm.SVC(gamma=0.001, C=100.)
    print(classifier)

    num_training = len(digits.data) - 5
    (X_training, ys_training), (X_test, ys_test) = \
        split_dataset(digits.data, digits.target, num_training)

    classifier.fit(X=X_training, y=ys_training)
    ys_pred = classifier.predict(X_test)
    print(ys_pred)

    for i_pred, (y_pred, y_test) in enumerate(zip(ys_pred, ys_test)):
        if y_pred == y_test:
            print('Test sample {i_sample} ({y_test}) is  CORRECTLY '
                  ' predicted as {y_pred}'
                  .format(y_pred=y_pred, i_sample=i_pred + 1, y_test=y_test))
        else:
            print('Test sample {i_sample} ({y_test}) is INCORRECTLY'
                  ' predicted as {y_pred}'
                  .format(y_pred=y_pred, i_sample=i_pred + 1, y_test=y_test))


def model_persistence():
    classifier = svm.SVC()

    iris = datasets.load_iris()
    (X_training, ys_training), (X_test, ys_test) = \
        split_dataset(iris.data, iris.target, num_training=len(iris.data) - 5)

    classifier.fit(X_training, ys_training)

    pkl_classifier = pickle.dumps(classifier)

    classifier2 = pickle.loads(pkl_classifier)
    ys_pred2 = classifier2.predict(X_test)

    ys_pred1 = classifier.predict(X_test)

    print ys_pred1 == ys_pred2
    print ys_pred1 == ys_test


def conventions():
    rng = np.random.RandomState(0)
    X1 = rng.rand(10, 2000)
    print(type(X1))
    print(X1.dtype)

    X2 = np.array(X1, dtype='float32')

    transformer = random_projection.GaussianRandomProjection()
    X_new = transformer.fit_transform(X2)
    print(X_new.dtype)

    # Field 'a' is little-endian int32, field 'b' is also little-endian int32
    # See https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html
    x = np.array([(1, 2), (3, 4)], dtype=[('a', '<i4'), ('b', '<i4')])
    print(x)
    print(x['a'])
    print(x.dtype)


def main():
    print('Learning and predicting')
    learn_and_predict()
    print('-' * 79 + '\n')

    print('Model persistence')
    model_persistence()
    print('-' * 79 + '\n')

    print('Conventions')
    conventions()
    print('-' * 79 + '\n')


if __name__ == '__main__':
    main()
