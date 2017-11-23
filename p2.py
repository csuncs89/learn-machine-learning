"""


"""
import pprint

import numpy as np

from sklearn import datasets, neighbors
import matplotlib
import matplotlib.pyplot as plt


def prepare_datasets():
    show_image = False

    print(matplotlib.__version__)

    iris = datasets.load_iris()
    data = iris.data
    # print(iris.DESCR)
    print(data.shape)

    digits = datasets.load_digits()
    print(digits.images.shape)

    pprint.pprint(dir(plt.cm))

    print(plt.cm.gray)
    print(plt.cm.gray_r)

    if show_image:
        plt.imshow(digits.images[-1], cmap=plt.cm.gray)
        plt.waitforbuttonpress()

        plt.imshow(digits.images[-1], cmap=plt.cm.gray_r)
        plt.waitforbuttonpress()

    num_images = digits.images.shape[0]
    data = digits.images.reshape((num_images, -1))
    print(digits.images.shape)
    print(data.shape)


def nearest_neighbor():
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target

    print(np.unique(iris_y))

    np.random.seed(0)
    print(np.random.permutation(10))
    print(np.random.permutation(10))
    np.random.seed(0)
    print(np.random.permutation(10))
    print(np.random.permutation(10))

    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    loc = -10
    iris_X_train = iris_X[indices[:loc]]
    iris_y_train = iris_y[indices[:loc]]
    iris_X_test = iris_X[indices[loc:]]
    iris_y_test = iris_y[indices[loc:]]

    knn = neighbors.KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)
    print(knn)
    print(knn.predict(iris_X_test))
    print(iris_y_test)


def main():
    prepare_datasets()
    nearest_neighbor()


if __name__ == '__main__':
    main()
