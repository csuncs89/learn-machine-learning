"""


"""
import pprint

from sklearn import datasets
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


def main():
    prepare_datasets()


if __name__ == '__main__':
    main()
