from __future__ import print_function
import time
import random
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import reuters
from keras.datasets import imdb
from keras.datasets import mnist
from keras.datasets import boston_housing
from keras.datasets import fashion_mnist
import hypothesis.strategies as st
from hypothesis import given

Values = st.integers(0, 5)


@given(v=Values)
def test_data(v):
    if (v == 0):
        (x_train, y_train), (x_test, y_test) = test_cifar()
        (x_train2, y_train2), (x_test2, y_test2) = test_cifar()
        assert len(x_train) == len(x_train2) == len(y_train) == len(y_train2)
        assert len(x_test) == len(x_test2) == len(y_test) == len(y_test2)
    elif (v == 1):
        (x_train, y_train), (x_test, y_test) = test_reuters()
        (x_train2, y_train2), (x_test2, y_test2) = test_reuters()
        assert len(x_train) == len(x_train2) == len(y_train) == len(y_train2)
        assert len(x_test) == len(x_test2) == len(y_test) == len(y_test2)
    elif (v == 2):
        (x_train, y_train), (x_test, y_test) = test_mnist()
        (x_train2, y_train2), (x_test2, y_test2) = test_mnist()
        assert len(x_train) == len(x_train2) == len(y_train) == len(y_train2)
        assert len(x_test) == len(x_test2) == len(y_test) == len(y_test2)
    elif (v == 3):
        (x_train, y_train), (x_test, y_test) = test_imdb()
        (x_train2, y_train2), (x_test2, y_test2) = test_imdb()
        assert len(x_train) == len(x_train2) == len(y_train) == len(y_train2)
        assert len(x_test) == len(x_test2) == len(y_test) == len(y_test2)
    elif (v == 4):
        (x_train, y_train), (x_test, y_test) = test_boston_housing()
        (x_train2, y_train2), (x_test2, y_test2) = test_boston_housing()
        assert len(x_train) == len(x_train2) == len(y_train) == len(y_train2)
        assert len(x_test) == len(x_test2) == len(y_test) == len(y_test2)
    else:
        (x_train, y_train), (x_test, y_test) = test_fashion_mnist()
        (x_train2, y_train2), (x_test2, y_test2) = test_fashion_mnist()
        assert len(x_train) == len(x_train2) == len(y_train) == len(y_train2)
        assert len(x_test) == len(x_test2) == len(y_test) == len(y_test2)


def test_cifar():
    random.seed(time.time())
    if random.random() > 0.33:
        return cifar10.load_data()
    elif random.random() > 0.66:
        return cifar100.load_data('fine')
    else:
        return cifar100.load_data('coarse')


def test_reuters():
    return reuters.load_data()


def test_mnist():
    return mnist.load_data()


def test_imdb():
    return imdb.load_data()


def test_boston_housing():
    return boston_housing.load_data()


def test_fashion_mnist():
    return fashion_mnist.load_data()
