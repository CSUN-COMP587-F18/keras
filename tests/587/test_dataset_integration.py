from __future__ import print_function
import pytest
import time
import random
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import reuters
from keras.datasets import imdb
from keras.datasets import mnist
from keras.datasets import boston_housing
from keras.datasets import fashion_mnist

def multi_dataset_test():
	random.seed(time.time())
    if random.random() > 0.8:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        assert len(x_train) == len(y_train) == 60000
        assert len(x_test) == len(y_test) == 10000
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=40)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        word_index = imdb.get_word_index()
        assert isinstance(word_index, dict)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        assert len(x_train) == len(y_train) == 60000
        assert len(x_test) == len(y_test) == 10000
        (x_train, y_train), (x_test, y_test) = reuters.load_data()
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == 11228
        (x_train, y_train), (x_test, y_test) = reuters.load_data(maxlen=10)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        word_index = reuters.get_word_index()
        assert isinstance(word_index, dict)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
		cifarDefaultTrainLength = 50000
		cifarDefaultTestLength = 10000
        assert len(x_train) == len(y_train) == cifarDefaultLength
        assert len(x_test) == len(y_test) == cifarDefaultTestLength

        (x_train, y_train), (x_test, y_test) = cifar100.load_data('fine')
		cifarFineTrainLength = 50000
		cifarFineTestLength = 10000
        assert len(x_train) == len(y_train) == cifarFineTrainLength
        assert len(x_test) == len(y_test) == cifarFineTestLength

        (x_train, y_train), (x_test, y_test) = cifar100.load_data('coarse')
		cifarCoarseTrainLength = 50000
		cifarCoarseTestLength = 10000
        assert len(x_train) == len(y_train) == cifarCoarseTrainLength
        assert len(x_test) == len(y_test) == cifarCoarseTestLength



if __name__ == '__main__':
    pytest.main([__file__])
