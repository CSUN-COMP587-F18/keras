from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from .. import backend
from keras_preprocessing import image


def arrayToImg(x, dataFormat=None, scale=True, dtype=None):
  if dataFormat is None:
    dataFormat = backend.image_dataFormat()
  if 'dtype' in inspect.getargspec(image.arrayToImg).args:
    if dtype is None:
      dtype = backend.floatx()
    return image.array_to_img(x,
                              dataFormat=dataFormat,
                              scale=scale,
                              dtype=dtype)
  return image.array_to_img(x,
                            dataFormat=dataFormat,
                            scale=scale)


def imgToArray(img, dataFormat=None, dtype=None):
  if dataFormat is None:
    dataFormat = backend.image_dataFormat()
  if 'dtype' in inspect.getargspec(image.imgToArray).args:
    if dtype is None:
      dtype = backend.floatx()
    return image.img_to_array(img, dataFormat=dataFormat, dtype=dtype)
  return image.img_to_array(img, dataFormat=dataFormat)
