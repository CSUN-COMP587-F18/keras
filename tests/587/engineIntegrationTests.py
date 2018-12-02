import threading
import pytest
import numpy as np
import pandas as pd
import sys

from keras import losses
from keras.engine import Input
from keras.engine.training import Model
from keras.engine import training_utils
from keras.layers import Dense, Dropout
from keras.utils.generic_utils import slice_arrays
from keras.models import Sequential
from keras.utils import Sequence


class RandSequence(Sequence):
    def __init__(self, batchSize, sequenceLength=12):
        self.batchSize = batchSize
        self.sequenceLength = sequenceLength
        self.logs = []

    def __len__(self):
        return self.sequenceLength

    def __getitem__(self, idx):
        self.logs.append(idx)
        return ([np.random.random((self.batchSize, 3)),
                 np.random.random((self.batchSize, 3))],
                [np.random.random((self.batchSize, 4)),
                 np.random.random((self.batchSize, 3))])

    def on_epoch_end(self):
        pass


def test_length_consistency():
    training_utils.check_array_length_consistency(None, None, None)
    a_np = np.random.random((4, 3, 3))
    training_utils.check_array_length_consistency(a_np, a_np, a_np)
    training_utils.check_array_length_consistency(
        [a_np, a_np], [a_np, a_np], [a_np, a_np])
    training_utils.check_array_length_consistency([None], [None], [None])

    b_np = np.random.random((3, 4))
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency(a_np, None, None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency(a_np, a_np, None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency([a_np], [None], None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency([a_np], [b_np], None)
    with pytest.raises(ValueError):
        training_utils.check_array_length_consistency([a_np], None, [b_np])


class threadsafe_iter:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def testslice_arrays():
    inputA = np.random.random((10, 3))
    slice_arrays(None)
    slice_arrays(inputA, 0)
    slice_arrays(inputA, 0, 1)
    slice_arrays(inputA, stop=2)
    inputA = [None, [1, 1], None, [1, 1]]
    slice_arrays(inputA, 0)
    slice_arrays(inputA, 0, 1)
    slice_arrays(inputA, stop=2)
    inputA = [None]
    slice_arrays(inputA, 0)
    slice_arrays(inputA, 0, 1)
    slice_arrays(inputA, stop=2)
    inputA = None
    slice_arrays(inputA, 0)
    slice_arrays(inputA, 0, 1)
    slice_arrays(inputA, stop=2)


@pytest.mark.skipif(sys.version_info < (3,),
                    reason='Cannot catch warnings in python 2')
def test_warnings():
    a = Input(shape=(3,), name='inputA')
    b = Input(shape=(3,), name='inputB')

    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)

    model = Model([a, b], [a_2, b_2])

    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model.compile(optimizer, loss, metrics=[], loss_weights=loss_weights,
                  sample_weight_mode=None)

    @threadsafe_generator
    def gen_data(batch_sz):
        while True:
            yield ([np.random.random((batch_sz, 3)),
                    np.random.random((batch_sz, 3))],
                   [np.random.random((batch_sz, 4)),
                    np.random.random((batch_sz, 3))])

    with pytest.warns(Warning) as w:
        out = model.fit_generator(gen_data(4),
                                  steps_per_epoch=10,
                                  use_multiprocessing=True,
                                  workers=2)
    warning_raised = any(['Sequence' in str(w_.message) for w_ in w])
    assert warning_raised, 'No warning raised when using generator with processes.'

    with pytest.warns(None) as w:
        out = model.fit_generator(RandSequence(3),
                                  steps_per_epoch=4,
                                  use_multiprocessing=True,
                                  workers=2)
    assert all(['Sequence' not in str(w_.message) for w_ in w]), (
        'A warning was raised for Sequence.')


def test_with_list_as_targets():
    model = Sequential()
    model.add(Dense(1, input_dim=3, trainable=False))
    model.compile('rmsprop', 'mse')

    x = np.random.random((2, 3))
    y = [0, 1]
    model.train_on_batch(x, y)


def test_check_not_failing():
    a = np.random.random((2, 1, 3))
    training_utils.check_loss_and_target_compatibility(
        [a], [losses.categorical_crossentropy], [a.shape])
    training_utils.check_loss_and_target_compatibility(
        [a], [losses.categorical_crossentropy], [(2, None, 3)])


def test_check_last_is_one():
    a = np.random.random((2, 3, 1))
    with pytest.raises(ValueError) as exc:
        training_utils.check_loss_and_target_compatibility(
            [a], [losses.categorical_crossentropy], [a.shape])

    assert 'You are passing a target array' in str(exc)


def test_check_bad_shape():
    a = np.random.random((2, 3, 5))
    with pytest.raises(ValueError) as exc:
        training_utils.check_loss_and_target_compatibility(
            [a], [losses.categorical_crossentropy], [(2, 3, 6)])

    assert 'targets to have the same shape' in str(exc)


def test_pd_df():  # testing dataframes via pandas
    inputA = Input(shape=(3,), name='inputA')
    inputB = Input(shape=(3,), name='inputB')

    x = Dense(4, name='dense_1')(inputA)
    y = Dense(3, name='desne_2')(inputB)

    model1 = Model(inputs=inputA, outputs=x)
    model2 = Model(inputs=[inputA, inputB], outputs=[x, y])

    optimizer = 'rmsprop'
    loss = 'mse'

    model1.compile(optimizer=optimizer, loss=loss)
    model2.compile(optimizer=optimizer, loss=loss)

    inputA_df = pd.DataFrame(np.random.random((10, 3)))
    inputB_df = pd.DataFrame(np.random.random((10, 3)))

    outputA_df = pd.DataFrame(np.random.random((10, 4)))
    outputB_df = pd.DataFrame(np.random.random((10, 3)))

    model1.fit(inputA_df,
               outputA_df)
    model2.fit([inputA_df, inputB_df],
               [outputA_df, outputB_df])
    model1.fit([inputA_df],
               [outputA_df])
    model1.fit({'inputA': inputA_df},
               outputA_df)
    model2.fit({'inputA': inputA_df, 'inputB': inputB_df},
               [outputA_df, outputB_df])

    model1.predict(inputA_df)
    model2.predict([inputA_df, inputB_df])
    model1.predict([inputA_df])
    model1.predict({'inputA': inputA_df})
    model2.predict({'inputA': inputA_df, 'inputB': inputB_df})

    model1.predict_on_batch(inputA_df)
    model2.predict_on_batch([inputA_df, inputB_df])
    model1.predict_on_batch([inputA_df])
    model1.predict_on_batch({'inputA': inputA_df})
    model2.predict_on_batch({'inputA': inputA_df, 'inputB': inputB_df})

    model1.evaluate(inputA_df,
                    outputA_df)
    model2.evaluate([inputA_df, inputB_df],
                    [outputA_df, outputB_df])
    model1.evaluate([inputA_df],
                    [outputA_df])
    model1.evaluate({'inputA': inputA_df},
                    outputA_df)
    model2.evaluate({'inputA': inputA_df, 'inputB': inputB_df},
                    [outputA_df, outputB_df])

    model1.train_on_batch(inputA_df,
                          outputA_df)
    model2.train_on_batch([inputA_df, inputB_df],
                          [outputA_df, outputB_df])
    model1.train_on_batch([inputA_df],
                          [outputA_df])
    model1.train_on_batch({'inputA': inputA_df},
                          outputA_df)
    model2.train_on_batch({'inputA': inputA_df, 'inputB': inputB_df},
                          [outputA_df, outputB_df])

    model1.test_on_batch(inputA_df,
                         outputA_df)
    model2.test_on_batch([inputA_df, inputB_df],
                         [outputA_df, outputB_df])
    model1.test_on_batch([inputA_df],
                         [outputA_df])
    model1.test_on_batch({'inputA': inputA_df},
                         outputA_df)
    model2.test_on_batch({'inputA': inputA_df, 'inputB': inputB_df},
                         [outputA_df, outputB_df])


if __name__ == '__main__':
    pytest.main([__file__])
