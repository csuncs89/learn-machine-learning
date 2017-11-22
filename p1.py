"""

"""
from sklearn import datasets
from sklearn import svm


def learn_and_predict():
    digits = datasets.load_digits()

    print(digits.data)

    classifier = svm.SVC(gamma=0.001, C=100.)
    print(classifier)

    num_training = len(digits.data) - 5
    X_training = digits.data[:num_training]
    ys_training = digits.target[:num_training]
    X_test = digits.data[num_training:]
    ys_test = digits.target[num_training:]

    classifier.fit(X=X_training, y=ys_training)
    ys_pred = classifier.predict(X_test)
    print(ys_pred)

    for i_pred, (y_pred, y_test) in enumerate(zip(ys_pred, ys_test)):
        if y_pred == y_test:
            print('Test sample {i_sample} ({y_test}) is  CORRECTLY '
                  ' predicted as {y_pred}'
                  .format(y_pred=y_pred, i_sample=i_pred + 1, y_test=y_test))
        else:
            print('Test sample {i_sample} ({y_test}) is INCORRECTLY'
                  ' predicted as {y_pred}'
                  .format(y_pred=y_pred, i_sample=i_pred + 1, y_test=y_test))


def main():
    learn_and_predict()


if __name__ == '__main__':
    main()
