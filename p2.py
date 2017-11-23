"""


"""
import pprint

import numpy as np

from sklearn import datasets, neighbors, linear_model, svm
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


def linear_regression():
    diabetes = datasets.load_diabetes()

    loc = -20
    diabetes_X_train = diabetes.data[:loc]
    diabetes_y_train = diabetes.target[:loc]
    diabetes_X_test = diabetes.data[loc:]
    diabetes_y_test = diabetes.target[loc:]

    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    print(regr.coef_)

    print(np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2))

    score = regr.score(diabetes_X_test, diabetes_y_test)
    print(score)


def linear_regression_ridge():
    show_image = False

    X = np.c_[0.5, 1].T
    print(X)
    y = [0.5, 1]

    test = np.c_[0, 2].T
    regr = linear_model.LinearRegression()
    plt.figure()

    np.random.seed(0)
    for __ in range(2):
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X
        regr.fit(this_X, y)
        print(test)
        print(regr.predict(test))
        plt.plot(test, regr.predict(test))
        plt.scatter(this_X, y, s=3)
        if show_image:
            plt.waitforbuttonpress()

    plt.close()

    regr1 = linear_model.Ridge(alpha=0.1)
    regr2 = linear_model.Ridge(alpha=1.0)
    plt.figure()
    np.random.seed(0)
    for __ in range(6):
        this_X = 0.1 * np.random.normal(size=(2, 1)) + X
        regr1.fit(this_X, y)
        regr2.fit(this_X, y)
        print(test)
        print(regr1.predict(test))
        plt.plot(test, regr1.predict(test))
        plt.scatter(this_X, y, s=3)

        if show_image:
            plt.waitforbuttonpress()

        print(test)
        print(regr2.predict(test))
        plt.plot(test, regr2.predict(test))
        plt.scatter(this_X, y, s=6)

        if show_image:
            plt.waitforbuttonpress()

    # bias/variance tradeoff
    # https://elitedatascience.com/bias-variance-tradeoff

    diabetes = datasets.load_diabetes()

    loc = -20
    diabetes_X_train = diabetes.data[:loc]
    diabetes_y_train = diabetes.target[:loc]
    diabetes_X_test = diabetes.data[loc:]
    diabetes_y_test = diabetes.target[loc:]

    alphas = np.logspace(-4, -1, 6)
    print(alphas)

    regr = linear_model.Ridge()
    scores = [regr.set_params(alpha=alpha)
              .fit(diabetes_X_train, diabetes_y_train)
              .score(diabetes_X_test, diabetes_y_test)
              for alpha in alphas]
    pprint.pprint(scores)


def linear_regression_sparse():
    diabetes = datasets.load_diabetes()
    loc = -20
    diabetes_X_train = diabetes.data[:loc]
    diabetes_y_train = diabetes.target[:loc]
    diabetes_X_test = diabetes.data[loc:]
    diabetes_y_test = diabetes.target[loc:]

    alphas = np.logspace(-4, -1, 6)
    regr = linear_model.Lasso()
    scores = [regr.set_params(alpha=alpha)
              .fit(diabetes_X_train, diabetes_y_train)
              .score(diabetes_X_test, diabetes_y_test)
              for alpha in alphas]
    best_alpha = alphas[np.argmax(scores)]
    print(max(scores))
    assert(alphas[np.argmax(scores)] == alphas[scores.index(max(scores))])
    print(best_alpha)
    regr.set_params(alpha=best_alpha).fit(diabetes_X_train, diabetes_y_train)
    print(regr)
    print(regr.coef_)
    print(regr.score(diabetes_X_test, diabetes_y_test))


def exercise_classify_digits():
    digits = datasets.load_digits()
    num_training = int(round(len(digits.data) * 0.9))
    X_train = digits.data[:num_training]
    y_train = digits.target[:num_training]
    X_test = digits.data[num_training:]
    y_test = digits.target[num_training:]

    knn = neighbors.KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print(knn)
    y_test_pred = knn.predict(X_test)
    print(list(y_test == y_test_pred).count(False))
    print(knn.score(X_test, y_test))

    logistic = linear_model.LogisticRegression()
    logistic.fit(X_train, y_train)
    y_test_pred = logistic.predict(X_test)
    print(list(y_test == y_test_pred).count(False))
    print(logistic.score(X_test, y_test))


def exercise_classify_digits_ans():
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target

    n_samples = len(X_digits)

    X_train = X_digits[:int(.9 * n_samples)]
    y_train = y_digits[:int(.9 * n_samples)]
    X_test = X_digits[int(.9 * n_samples):]
    y_test = y_digits[int(.9 * n_samples):]

    knn = neighbors.KNeighborsClassifier()
    logistic = linear_model.LogisticRegression()

    print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
    print('LogisticRegression score: %f'
          % logistic.fit(X_train, y_train).score(X_test, y_test))


def exercise_svm():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 0, :2]
    y = y[y != 0]

    np.random.seed(0)
    indices = np.random.permutation(len(X))
    loc = int(len(X) * 0.9)
    X_train = X[indices[:loc]]
    y_train = y[indices[:loc]]
    X_test = X[indices[loc:]]
    y_test = y[indices[loc:]]

    for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
        classifier = svm.SVC(kernel=kernel, gamma=10)
        classifier.fit(X_train, y_train)

        plt.figure(fig_num)
        plt.clf()
        plt.scatter(x=X[:, 0], y=X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)
        plt.scatter(x=X_test[:, 0], y=X_test[:, 1], s=80, facecolors='none',
                    zorder=10, edgecolor='k')

        plt.axis('tight')
        axis0_min = X[:, 0].min()
        axis0_max = X[:, 0].max()
        axis1_min = X[:, 1].min()
        axis1_max = X[:, 1].max()

        # http://louistiao.me/posts/numpy-mgrid-vs-meshgrid/
        XX, YY = np.mgrid[axis0_min:axis0_max:200j,
                          axis1_min:axis1_max:200j]
        print(XX.shape, YY.shape)
        print(np.c_[XX.ravel(), YY.ravel()].shape)

        Z = classifier.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)

        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-0.5, 0, 0.5])
        plt.title(kernel)

    plt.show()


def exercise_svm_ans():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = X[y != 0, :2]
    y = y[y != 0]

    n_sample = len(X)

    np.random.seed(0)
    order = np.random.permutation(n_sample)
    X = X[order]
    y = y[order].astype(np.float)

    X_train = X[:int(.9 * n_sample)]
    y_train = y[:int(.9 * n_sample)]
    X_test = X[int(.9 * n_sample):]
    y_test = y[int(.9 * n_sample):]

    # fit the model
    for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
        clf = svm.SVC(kernel=kernel, gamma=10)
        clf.fit(X_train, y_train)

        plt.figure(fig_num)
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)

        # Circle out the test data
        plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                    zorder=10, edgecolor='k')

        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

        plt.title(kernel)
    plt.show()


def main():
    print('Dataset')
    prepare_datasets()
    print('-' * 79 + '\n')

    print('Nearest neighbor')
    nearest_neighbor()
    print('-' * 79 + '\n')

    print('Linear regression')
    linear_regression()
    print('-' * 79 + '\n')

    print('Ridge linear regression')
    linear_regression_ridge()
    print('-' * 79 + '\n')

    print('Sparse linear regression by Lasso method')
    linear_regression_sparse()
    print('-' * 79 + '\n')

    print('Exercise - classify digits')
    exercise_classify_digits()
    print('-' * 79 + '\n')

    print('Exercise - classify digits (answer)')
    exercise_classify_digits_ans()
    print('-' * 79 + '\n')

    print('Exercise - SVM')
    exercise_svm()
    print('-' * 79 + '\n')

    print('Exercise - SVM (answer)')
    exercise_svm_ans()
    print('-' * 79 + '\n')


if __name__ == '__main__':
    main()
