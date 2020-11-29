#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

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
    w = [0 for i in range(784)]
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
    return np.randint(0, i)


def SGD_hinge_step(before_w, n0, t, C, xi, yi,):
    nt = n0/t
    after_w = (1-nt)*before_w  # init
    if(yi*before_w*xi < 1):
        after_w = after_w + nt*C*yi*xi
    return after_w
