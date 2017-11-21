"""

"""
from sklearn import datasets
from sklearn import svm


def main():
    iris = datasets.load_iris()
    digits = datasets.load_digits()

    print(digits.data)

    classifier = svm.SVC(gamma=0.001, C=100.)
    print(classifier)

    num_training = len(digits.data) - 1
    X_training = digits.data[:num_training]
    ys_training = digits.target[:num_training]
    X_test = digits.data[num_training:]
    ys_test = digits.target[num_training:]

    classifier.fit(X=X_training, y=ys_training)
    ys_pred = classifier.predict(X_test)
    print(ys_pred)

    for i_pred, (y_pred, y_test) in enumerate(zip(ys_pred, ys_test)):
        if y_pred == y_test:
            print('  Correct prediction (%d) for sample %d (%d)' %
                  (y_pred, i_pred + 1, y_test))
        else:
            print('Incorrect prediction (%d) for sample %d (%d)' %
                  (y_pred, i_pred + 1, y_test))


if __name__ == '__main__':
    main()
