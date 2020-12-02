#################################
# Your name: Bar Yaacovi 208939009
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

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
    number_of_samples = len(data)
    w_l_vectors = np.zeros((10,784),dtype=np.longfloat)
    for t in range(1, T+1):
        random_i = random_sample_from_i(number_of_samples)
        xi, yi = data[random_i], labels[random_i]
        w_l_vectors = SGD_ce_step(w_l_vectors, eta_0, t, xi, yi)
    return w_l_vectors

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

def SGD_ce_step(before_w_l_vectors, n0, t, xi, yi,):
    nt = n0
    y_values = range(10)
    softmax_probabilities = compute_softmax_vector(before_w_l_vectors,xi,y_values)
    after_w_l_vectors = np.zeros((len(before_w_l_vectors), len(xi)))
    for k in range(len(before_w_l_vectors)):
        if (k == int(yi)):
            after_w_l_vectors[k] = before_w_l_vectors[k] - nt * (softmax_probabilities[k]- 1)*xi
        else:
            after_w_l_vectors[k] = before_w_l_vectors[k] - nt * (softmax_probabilities[k])*xi
    return after_w_l_vectors


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


def compute_softmax_vector(predictor_l_vectors,xi,y_values):
    sum = 0
    probabilities = [0 for i in range(len(predictor_l_vectors))]
    exp_values = np.array([np.inner(predictor_l_vectors[j], xi) for j in range(len(predictor_l_vectors))])
    argmax_exp = exp_values.argmax()
    normalize_power = np.inner(predictor_l_vectors[argmax_exp], xi)
    for i in range(len(predictor_l_vectors)):
        w_for_i = predictor_l_vectors[i]
        sum = sum + (np.exp(np.inner(w_for_i, xi) - normalize_power))
    for i in range(len(predictor_l_vectors)):
        w_for_i = predictor_l_vectors[i]
        nominator = np.exp(np.inner(w_for_i, xi) - normalize_power) 
        probabilities[i] = nominator / sum
    return probabilities

def predict_ce_softmax(predictor_l_vectors, xi, y_values):
    probabilities = compute_softmax_vector(predictor_l_vectors,xi,y_values)
    return np.random.choice(y_values, p=probabilities)
    


def cross_validation_ce(validation_data, validation_labels, predictor_l_vectors, y_values):
    worked = 0
    for i in range(len(validation_data)):
        x, y = validation_data[i], validation_labels[i]
        predicted_y = predict_ce_softmax(predictor_l_vectors,x, y_values)
        if(int(y) == predicted_y):
            worked = worked + 1
    return worked/len(validation_data)


def q1_a():
    T = 1000
    C = 1
    n0_candidates = np.array([pow(10, i)
                              for i in range(-5, 4)], dtype=np.longfloat)
    n0_emp_average = np.array([0 for i in range(-5, 4)], dtype=np.longfloat)
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
    line, = plt.plot(n0_candidates, n0_emp_average, label="eta0 to accuracy")
    plt.legend()
    plt.xlim(pow(10, -5), 10**3)
    plt.xscale('log')
    #plt.show()
    return n0_candidates[n0_emp_average.argmin()]


def q1_b():
    T = 1000
    n0 = 1
    C_candidates = np.array([pow(10, i) for i in range(-5, 4)])
    C_emp_average = np.array([0 for i in range(-5, 4)], dtype=np.float64)
    number_of_runs = 10
    for i in range(number_of_runs):
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
        for j in range(len(C_candidates)):
            predictor = SGD_hinge(train_data, train_labels,
                                  C_candidates[j], n0, T)
            C_emp_average[j] = C_emp_average[j] + \
                cross_validation(
                    validation_data, validation_labels, predictor)/number_of_runs
    print(C_emp_average)
    line, = plt.plot(C_candidates, C_emp_average,
                     label="C candidates to accuracy")
    plt.legend()
    plt.xlim(pow(10, -5), 10**3)
    plt.xscale('log')
    #plt.show()
    return C_candidates[C_emp_average.argmin()]


def q1_c_d():
    T = 20000
    n0 = 1
    C = 10 ** -4
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    predictor = SGD_hinge(train_data, train_labels, C, n0, T)
    predictor_2d = np.reshape(predictor, (28, 28))
    plt.imshow(predictor_2d)
    #plt.show()
    accuracy = cross_validation(test_data, test_labels, predictor)
    print(accuracy)


def q2_a():
    T = 1000
    n0_candidates = np.array([pow(10, i)
                              for i in range(-5, 6)], dtype=np.longfloat)
    n0_emp_average = np.array([0 for i in range(-5, 6)], dtype=np.longfloat)
    number_of_runs = 10
    for i in range(number_of_runs):
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
        for j in range(len(n0_candidates)):
            predictor = SGD_ce(train_data, train_labels,
                                  n0_candidates[j], T)
            n0_emp_average[j] = n0_emp_average[j] + \
                cross_validation_ce(
                    validation_data, validation_labels, predictor, range(10))/number_of_runs
    print(n0_emp_average)
    line, = plt.plot(n0_candidates, n0_emp_average, label="eta0 to accuracy")
    plt.legend()
    plt.xlim(pow(10, -5), 10**5)
    plt.xscale('log')
    #plt.show()
    return n0_candidates[n0_emp_average.argmin()]


def q2_b_c():
    T = 20000
    n0 = 10 ** -3
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    accuracy = 0
    predictor = SGD_ce(train_data, train_labels,n0, T)
    for i in range(10):
        predictor_2d = np.reshape(predictor[i], (28, 28))
        plt.title("w{} image:".format(i))
        plt.imshow(predictor_2d)
        #plt.show()
    accuracy =  + cross_validation_ce(validation_data, validation_labels, predictor, range(10))
    print(accuracy)
    return accuracy
