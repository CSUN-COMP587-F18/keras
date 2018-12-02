import pytest
import numpy as np

from keras.utils.test_utils import layer_test
from keras import layers


def test_flatten():

    def test_4d():
        np_inp_channels_last = np.arange(24, dtype='float32').reshape((1, 4, 3, 2))

        np_output_cl = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_last'},
                                  input_data=np_inp_channels_last)

        np_inp_channels_first = np.transpose(np_inp_channels_last,
                                             [0, 3, 1, 2])

        np_output_cf = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_first'},
                                  input_data=np_inp_channels_first,
                                  expected_output=np_output_cl)

    def test_3d():
        np_inp_channels_last = np.arange(12, dtype='float32').reshape(
            (1, 4, 3))

        np_output_cl = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_last'},
                                  input_data=np_inp_channels_last)

        np_inp_channels_first = np.transpose(np_inp_channels_last,
                                             [0, 2, 1])

        np_output_cf = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_first'},
                                  input_data=np_inp_channels_first,
                                  expected_output=np_output_cl)

    def test_5d():
        np_inp_channels_last = np.arange(120, dtype='float32').reshape(
            (1, 5, 4, 3, 2))

        np_output_cl = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_last'},
                                  input_data=np_inp_channels_last)

        np_inp_channels_first = np.transpose(np_inp_channels_last,
                                             [0, 4, 1, 2, 3])

        np_output_cf = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_first'},
                                  input_data=np_inp_channels_first,
                                  expected_output=np_output_cl)

    test_3d()
    test_4d()
    test_5d()


if __name__ == '__main__':
    pytest.main([__file__])
