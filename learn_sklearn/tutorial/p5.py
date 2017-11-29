"""

"""
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn import linear_model
from sklearn import model_selection
from sklearn import pipeline
import numpy as np

import deco_utils


@deco_utils.print_banner('Pipelining PCA and logistic regression')
def pipelining():
    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()

    pipe = pipeline.Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target

    pca.fit(X_digits)

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([0.2, 0.2, 0.7, 0.7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)

    model_selection
    estimator = model_selection.GridSearchCV(
        pipe, dict(pca__n_components=n_components, logistic__C=Cs))

    estimator.fit(X_digits, y_digits)

    for param, score in zip(estimator.cv_results_['params'],
                            estimator.cv_results_['mean_test_score']):
        print('{param}: {score}'.format(param=param, score=score))

    print(estimator.best_estimator_)
    print(estimator.best_estimator_.named_steps['pca'].n_components)
    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))

    plt.show()


@deco_utils.print_banner('Face recognition with eigenfaces')
def face_recognition():
    lfw_peope = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_peope.images.shape
    print('n_samples={0} h={1} w={2}'.format(n_samples, h, w))


def main():
    # pipelining()
    face_recognition()

if __name__ == '__main__':
    main()
