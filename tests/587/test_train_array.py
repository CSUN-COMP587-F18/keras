import pytest
import numpy as np


from keras.engine import training_utils
from keras.utils.generic_utils import slice_arrays


def test_check_array_length_consistency():
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


def testslice_arrays():
    input_a = np.random.random((10, 3))
    slice_arrays(None)
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)
    input_a = [None, [1, 1], None, [1, 1]]
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)
    input_a = [None]
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)
    input_a = None
    slice_arrays(input_a, 0)
    slice_arrays(input_a, 0, 1)
    slice_arrays(input_a, stop=2)
