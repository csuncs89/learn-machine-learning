"""

"""
import pprint

from sklearn import cluster
from sklearn import datasets
from sklearn import decomposition
from sklearn.feature_extraction.image import grid_to_graph
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.signal

import deco_utils


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

    '''
    012
    345
    678
    (0, 0), (0, 1), (0, 3)
    (1, 1), (1, 0), (1, 2), (1, 4)
    (2, 2), (2, 1), (2, 5)
    ...
    '''
    connectivity = grid_to_graph(3, 3)
    print(connectivity.toarray())

    connectivity = grid_to_graph(*images[0].shape)

    agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                         n_clusters=10)
    print(X[0])
    print(X[1])
    print(X.shape)

    agglo.fit(X)
    X_reduced = agglo.transform(X)
    print(X_reduced[0])
    print(X_reduced[1])
    print(X_reduced.shape)

    agglo.fit(X[:10])
    X_reduced = agglo.transform(X[:10])
    print(X_reduced[0])
    print(X_reduced[1])
    print(X_reduced.shape)


@deco_utils.print_banner('Principle component analysis: PCA')
def principle_component_analysis():
    x1 = np.random.normal(size=100)
    x2 = np.random.normal(size=100)
    x3 = x1 + x2
    X = np.c_[x1, x2, x3]

    pca = decomposition.PCA()
    pca.fit(X)
    print(pca.explained_variance_)

    pca.set_params(n_components=2)
    X_reduced = pca.fit_transform(X)
    print(X_reduced.shape)


@deco_utils.print_banner('Independent component analysis: ICA')
def independent_component_analysis():
    t = np.linspace(0, 10, 2000)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))
    s3 = scipy.signal.sawtooth(2 * np.pi * t)

    print(s1.shape)
    print(s2.shape)
    print(s3.shape)

    S = np.c_[s1, s2, s3]
    print(S.shape)

    S += 0.2 * np.random.normal(size=S.shape)
    S /= S.std(axis=0)
    print(S.shape)

    A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
    print(A.shape)
    X = np.dot(S, A.T)
    print(X.shape)

    ica = decomposition.FastICA()
    S_ = ica.fit_transform(X)
    A_ = ica.mixing_.T

    print(np.allclose(X, np.dot(S_, A_) + ica.mean_))


def main():
    kmeans_clustering()
    vector_quantization()
    hierarchical_clustering()
    principle_component_analysis()
    independent_component_analysis()

if __name__ == '__main__':
    main()
