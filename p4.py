"""

"""
import pprint

from sklearn import cluster
from sklearn import datasets

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


def main():
    kmeans_clustering()

if __name__ == '__main__':
    main()
