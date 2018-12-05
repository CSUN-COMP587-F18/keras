import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.applications import imagenet_utils as utils


def test_preprocess_input():
    randomInteger = np.random.uniform(0, 255, (2, 10, 10, 3))
    randomInteger32Bit = randomInteger.astype('int32')
    assert utils.preprocess_input(randomInteger).shape == randomInteger.shape
    assert utils.preprocess_input(randomInteger32Bit).shape == randomInteger32Bit.shape

    preprocessOutput1 = utils.preprocess_input(randomInteger, 'channels_last')
    preprocessOutput32Bit1 = utils.preprocess_input(randomInteger32Bit, 'channels_last')
    preprocessOutputRandTrans1 = utils.preprocess_input(np.transpose(randomInteger, (0, 3, 1, 2)),
                                  'channels_first')
    preprocessOutputRandTrans2 = utils.preprocess_input(np.transpose(randomInteger32Bit, (0, 3, 1, 2)),
                                     'channels_first')
    assert_allclose(preprocessOutput1, preprocessOutputRandTrans1.transpose(0, 2, 3, 1))
    assert_allclose(preprocessOutput32Bit1, preprocessOutputRandTrans2.transpose(0, 2, 3, 1))

    # testing single image
    randomInteger = np.random.uniform(0, 255, (10, 10, 3))
    randomInteger32Bit = randomInteger.astype('int32')
    assert utils.preprocess_input(randomInteger).shape == randomInteger.shape
    assert utils.preprocess_input(randomInteger32Bit).shape == randomInteger32Bit.shape

    preprocessOutput1 = utils.preprocess_input(randomInteger, 'channels_last')
    preprocessOutput32Bit1 = utils.preprocess_input(randomInteger32Bit, 'channels_last')
    preprocessOutputRandTrans1 = utils.preprocess_input(np.transpose(randomInteger, (2, 0, 1)),
                                  'channels_first')
    preprocessOutputRandTrans2 = utils.preprocess_input(np.transpose(randomInteger32Bit, (2, 0, 1)),
                                     'channels_first')
    assert_allclose(preprocessOutput1, preprocessOutputRandTrans1.transpose(1, 2, 0))
    assert_allclose(preprocessOutput32Bit1, preprocessOutputRandTrans2.transpose(1, 2, 0))

    # test writing over data
    for mode in ['torch', 'tf']:
        randomInteger = np.random.uniform(0, 255, (2, 10, 10, 3))
        randomInteger32Bit = randomInteger.astype('int')
        randomPreprocessOutput = utils.preprocess_input(randomInteger, mode=mode)
        randomPreprocessOutput32Bit = utils.preprocess_input(randomInteger32Bit)
        assert_allclose(randomInteger, randomPreprocessOutput)
        assert randomInteger32Bit.astype('float').max() != randomPreprocessOutput32Bit.max()
    # caffe mode is different
    randomInteger = np.random.uniform(0, 255, (2, 10, 10, 3))
    randomInteger32Bit = randomInteger.astype('int')
    randomPreprocessOutput = utils.preprocess_input(randomInteger, data_format='channels_last', mode='caffe')
    randomPreprocessOutput32Bit = utils.preprocess_input(randomInteger32Bit)
    assert_allclose(randomInteger, randomPreprocessOutput[..., ::-1])
    assert randomInteger32Bit.astype('float').max() != randomPreprocessOutput32Bit.max()


if __name__ == '__main__':
    pytest.main([__file__])
