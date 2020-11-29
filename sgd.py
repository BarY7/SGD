#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import warnings
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(
        train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(
        validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(
        test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(
        np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000+test_idx, :].astype(float)
    test_labels = labels[8000+test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(
        train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(
        validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(
        test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    number_of_samples = len(data)
    w = np.array([0 for i in range(784)], dtype=np.longfloat)
    for t in range(1, T+1):
        random_i = random_sample_from_i(number_of_samples)
        xi, yi = data[random_i], labels[random_i]
        w = SGD_hinge_step(w, eta_0, t, C, xi, yi)
    return w


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    # TODO: Implement me
    pass

#################################

# Place for additional code

#################################

# does a single stop on before_w(wt) and returns
# wt+1


def random_sample_from_i(i):
    return np.random.randint(0, i)


def SGD_hinge_step(before_w, n0, t, C, xi, yi,):
    nt = n0/t
    after_w = None
    if(yi*np.inner(before_w, xi) < 1):
        after_w = (1-nt)*before_w + nt*C*yi*xi
    else:
        after_w = np.multiply((1-nt), before_w)

    return after_w


def cross_validation(validation_data, validation_labels, predictor):
    worked = 0
    for i in range(len(validation_data)):
        x, y = validation_data[i], validation_labels[i]
        predicted_y = -1
        if(np.inner(x, predictor) >= 0):
            predicted_y = 1
        if(y == predicted_y):
            worked = worked + 1
    return worked/len(validation_data)


def q1_a():
    T = 1000
    C = 1
    n0_candidates = np.array([pow(10, i)
                              for i in range(-5, 6)], dtype=np.longfloat)
    n0_emp_average = np.array([0 for i in range(-5, 6)], dtype=np.longfloat)
    number_of_runs = 10
    for i in range(number_of_runs):
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
        for j in range(len(n0_candidates)):
            predictor = SGD_hinge(train_data, train_labels,
                                  C, n0_candidates[j], T)
            n0_emp_average[j] = n0_emp_average[j] + \
                cross_validation(
                    validation_data, validation_labels, predictor)/number_of_runs
    print(n0_emp_average)
    plt.plot(n0_candidates, n0_emp_average)
    plt.xlim(10**-5, 10**3)
    plt.show()
    return n0_candidates[n0_emp_average.argmin()]


def q1_b():
    T = 1000
    n0 = 1
    C_candidates = np.array([pow(10, -i) for i in range(-5, 5)])
    C_emp_average = np.array([0 for i in range(-5, 5)])
    number_of_runs = 10
    for i in range(number_of_runs):
        for j in range(len(C_candidates)):
            train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
            predictor = SGD_hinge(train_data, train_labels,
                                  C_candidates[j], n0, T)
            C_emp_average[j] = C_emp_average[j] + \
                cross_validation(
                    validation_data, validation_labels, predictor)/number_of_runs
    print(C_emp_average)
    plt.plot(C_candidates, C_emp_average)
    return C_candidates[C_emp_average.argmin()]


def q1_c_d():
    T = 20000
    n0 =1
    C = 4
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    predictor = SGD_hinge(train_data, train_labels, C, n0, T)
    predictor_2d = np.reshape(predictor, (-1, 2))
    plt.imshow(predictor_2d)
    accuracy = cross_validation(test_data, test_labels, predictor_2d)
    print(accuracy)


q1_a()
