import pytest
import os
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from keras.models import Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import save_model, load_model


def testModelModularity():
    testModel = Sequential()
    testModel.add(Dense(2, input_shape=(3,)))
    testModel.add(RepeatVector(3))
    testModel.add(TimeDistributed(Dense(3)))
    testModel.compile(loss=losses.MSE,
                      optimizer=optimizers.RMSprop(lr=0.0001),
                      metrics=[metrics.categorical_accuracy],
                      sample_weight_mode='temporal')
    rand1 = np.random.random((1, 3))
    rand2 = np.random.random((1, 3, 3))
    testModel.train_on_batch(rand1, rand2)

    out = testModel.predict(rand1)
    _, fname = tempfile.mkstemp('.h5')
    save_model(testModel, fname)

    new_model = load_model(fname)
    os.remove(fname)

    out2 = new_model.predict(rand1)
    assert_allclose(out, out2, atol=1e-05)

    rand1 = np.random.random((1, 3))
    rand2 = np.random.random((1, 3, 3))
    testModel.train_on_batch(rand1, rand2)
    new_model.train_on_batch(rand1, rand2)
    out = testModel.predict(rand1)
    out2 = new_model.predict(rand1)
    assert_allclose(out, out2, atol=1e-05)


if __name__ == '__main__':
    pytest.main([__file__])
