"""

"""
import pprint

from sklearn import cluster
from sklearn import datasets
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

import deco_utils
from sklearn.feature_extraction.image import grid_to_graph


@deco_utils.print_banner('KMeans clustering')
def kmeans_clustering():
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(X_iris)

    print(kmeans)
    pprint.pprint(dir(kmeans))
    print(kmeans.labels_)
    print(kmeans.labels_[::10])
    print(y_iris[::10])


@deco_utils.print_banner('Vector quantization')
def vector_quantization():
    face = scipy.misc.face(gray=True)
    print(face)
    print(type(face))
    print(face.shape)

    plt.figure(1)
    plt.imshow(face, cmap=plt.cm.gray)

    X = face.reshape((-1, 1))

    k_means = cluster.KMeans(n_clusters=5, n_init=1)
    k_means.fit(X)
    print(k_means)

    print(k_means.cluster_centers_)
    values = k_means.cluster_centers_.squeeze()
    print(values)
    labels = k_means.labels_
    print(labels)
    face_compressed = np.choose(labels, values).reshape(face.shape)
    plt.figure(2)
    plt.imshow(face_compressed, cmap=plt.cm.gray)

    plt.show()


def print_args(*args, **kwargs):
    print('args: {0}'.format(args))
    print('kwargs: {0}'.format(kwargs))
    
def print_args2(nx, ny):
    print('{0}, {1}'.format(nx, ny))

@deco_utils.print_banner('Feature agglomeration')
def hierarchical_clustering():
    digits = datasets.load_digits()
    images = digits.images
    print(images.shape)
    X = np.reshape(images, (len(images), -1))
    print(X.shape)
    print_args(*images[0].shape)
    print_args2(*images[0].shape)
    connectivity = grid_to_graph(8, 8)
    
    print(connectivity.toarray())


def main():
    # kmeans_clustering()
    # vector_quantization()
    hierarchical_clustering()

if __name__ == '__main__':
    main()
