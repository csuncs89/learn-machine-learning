"""

"""
import pprint

from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

from lml.utils import deco_utils


@deco_utils.print_banner('Print score')
def f1():
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target

    svc = svm.SVC(kernel='linear')

    loc = -100
    X_train, y_train = X_digits[:loc], y_digits[:loc]
    X_test, y_test = X_digits[loc:], y_digits[loc:]

    score = svc.fit(X_train, y_train).score(X_test, y_test)

    print(score)


@deco_utils.print_banner('Cross validation scores')
def f2():
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    X_folds = np.array_split(X_digits, 3)
    y_folds = np.array_split(y_digits, 3)

    scores = []

    svc = svm.SVC(kernel='poly')

    for k in range(3):
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)

        scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
    print(scores)

    k_fold = model_selection.KFold(n_splits=3)
    scores = [svc.fit(X_digits[train], y_digits[train])
              .score(X_digits[test], y_digits[test])
              for train, test in k_fold.split(X_digits)]
    print(scores)

    scores = model_selection.cross_val_score(
        svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
    print(scores)

    scores = model_selection.cross_val_score(
        svc, X_digits, y_digits, cv=k_fold, n_jobs=-1,
        scoring='precision_macro')
    print(scores)


@deco_utils.print_banner('Exercise - cross validation of digits dataset')
def exercise_digits():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    svc = svm.SVC(kernel='poly')
    C_s = np.logspace(-10, 0, 10)
    print(C_s)

    scores = list()
    scores_std = list()
    for C in C_s:
        svc.set_params(C=C)
        this_scores = model_selection.cross_val_score(svc, X, y, n_jobs=1)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))

    pprint.pprint(scores)
    pprint.pprint(scores_std)

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.semilogx(C_s, scores)
    plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
    plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    print(locs, labels)
    plt.yticks(locs, list(map(lambda x: '%g' % x, locs)))
    plt.ylabel('CV score')
    plt.xlabel('Parameter C')
    plt.ylim(0, 1.1)
    plt.show()


@deco_utils.print_banner('Grid search')
def grid_search():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    C_s = np.logspace(-2, 6, 5)
    gamma_s = np.logspace(-8, 0, 5)
    print(C_s)
    print(gamma_s)
    svc = svm.SVC(kernel='rbf')
    clf = model_selection.GridSearchCV(estimator=svc,
                                       param_grid=dict(C=C_s, gamma=gamma_s),
                                       n_jobs=-1)
    clf.fit(X[:1000], y[:1000])
    print('Best score: {score}'.format(score=clf.best_score_))
    print('Best C: {C}'.format(C=clf.best_estimator_.C))
    print('Best gamma: {gamma}'.format(gamma=clf.best_estimator_.gamma))


@deco_utils.print_banner('Exercise - grid search of diabetes dataset')
def exercise_diabetes():
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]

    lasso = linear_model.Lasso(random_state=0)
    alphas = np.logspace(-4, -0.5, 30)

    n_folds = 3
    clf = model_selection.GridSearchCV(
        estimator=lasso, param_grid=[{'alpha': alphas}],
        cv=n_folds, refit=False)
    clf.fit(X, y)

    scores = clf.cv_results_['mean_test_score']

    scores_std = clf.cv_results_['std_test_score']

    print(clf.cv_results_['std_test_score'])

    print(np.std([clf.cv_results_['split0_test_score'][0],
                  clf.cv_results_['split1_test_score'][0],
                  clf.cv_results_['split2_test_score'][0]]))

    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, scores)
    plt.semilogx(alphas, scores + scores_std, 'b--')
    plt.semilogx(alphas, scores - scores_std, 'b--')

    plt.fill_between(alphas, scores + scores_std, scores - scores_std,
                     alpha=0.2)

    plt.ylabel('CV score +/- std error')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([alphas[0], alphas[-1]])
    plt.show()

    lasso_cv = linear_model.LassoCV(alphas=alphas, random_state=0)
    k_fold = model_selection.KFold(3)

    for k, (train, test) in enumerate(k_fold.split(X, y)):
        lasso_cv.fit(X[train], y[train])
        print('[fold {0}] alpha: {1:.5f}, score: {2:.5f}'
              .format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))


def main():
    f1()
    f2()
    exercise_digits()
    grid_search()
    exercise_diabetes()


if __name__ == '__main__':
    main()
