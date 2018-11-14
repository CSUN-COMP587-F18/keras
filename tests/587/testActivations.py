import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras import activations


def getStdValues():  # set of floats for testing activations
    return np.array([[0, 0.1, 0.5, 0.9, 1.0]], dtype=K.floatx())


def referenceSigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1 + z)


def testSerialization():
    all_activations = ['softmax', 'relu', 'elu', 'tanh',
                       'sigmoid', 'hard_sigmoid', 'linear',
                       'softplus', 'softsign', 'selu']
    for name in all_activations:
        fn = activations.get(name)
        ref_fn = getattr(activations, name)
        assert fn == ref_fn
        config = activations.serialize(fn)
        fn = activations.deserialize(config)
        assert fn == ref_fn


def testSigmoid():  # testing sigmoid implementation

    sigmoid = np.vectorize(referenceSigmoid)

    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.sigmoid(x)])
    testValues = getStdValues()

    result = f([testValues])[0]
    expectedResult = sigmoid(testValues)
    assert_allclose(result, expectedResult, rtol=1e-05)


def testRelu():  # testing rectified linear units
    x = K.placeholder(ndim=2)
    f = K.function([x], [activations.relu(x)])

    testValues = getStdValues()
    result = f([testValues])[0]
    assert_allclose(result, testValues, rtol=1e-05)

    # testing max_value
    testValues = np.array([[0.5, 1.5]], dtype=K.floatx())
    f = K.function([x], [activations.relu(x, max_value=1.)])
    result = f([testValues])[0]
    assert np.max(result) <= 1.

    # testing max_value == 6.
    testValues = np.array([[0.5, 6.]], dtype=K.floatx())
    f = K.function([x], [activations.relu(x, max_value=1.)])
    result = f([testValues])[0]
    assert np.max(result) <= 6.


def testLinearity():
    testValues = [1, 5, True, None]
    for x in testValues:
        assert(x == activations.linear(x))


if __name__ == '__main__':
    pytest.main([__file__])
