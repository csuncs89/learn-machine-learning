"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from lml.utils import deco_utils


@deco_utils.print_banner('Ordinary Least Squares')
def ordinary_least_squares():
    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit([[0, 0], [1, 1], [2, 2]], [2, 3, 4])
    print(reg)
    print(reg.coef_)
    print(reg.intercept_)


@deco_utils.print_banner('Linear Regression Example')
def linear_regression_example():
    diabetes = datasets.load_diabetes()

    print(diabetes.data.shape)

    diabetes_X = diabetes.data[:, np.newaxis, 2]
    print(diabetes_X.shape)

    loc = -20
    diabetes_X_train = diabetes_X[:loc]
    diabetes_X_test = diabetes_X[loc:]

    diabetes_y_train = diabetes.target[:loc]
    diabetes_y_test = diabetes.target[loc:]

    regr = linear_model.LinearRegression()

    regr.fit(diabetes_X_train, diabetes_y_train)

    diabetes_y_pred = regr.predict(diabetes_X_test)

    print('Coefficients: \n', regr.coef_)

    print('Mean squared error: {0:.2f}'
          .format(metrics.mean_squared_error(diabetes_y_test,
                                             diabetes_y_pred)))
    print('Variance score: {0:.2f}'
          .format(metrics.r2_score(diabetes_y_test, diabetes_y_pred)))

    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    # plt.xticks(())
    plt.yticks(())

    plt.show()


def main():
    ordinary_least_squares()
    linear_regression_example()

if __name__ == '__main__':
    main()
