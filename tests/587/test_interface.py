from __future__ import print_function

import pytest

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb


def test_interface():
    maxFeat = 5000
    maxLength = 400
    batchSize = 32
    embed = 50
    fils = 250
    kernSize = 3
    hidDimensions = 250
    epochs = 2

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxFeat)

    x_train = sequence.pad_sequences(x_train, maxlen=maxLength)
    x_test = sequence.pad_sequences(x_test, maxlen=maxLength)

    model = Sequential()

    model.add(Embedding(maxFeat,
                        embed,
                        input_length=maxLength))
    model.add(Dropout(0.2))

    model.add(Conv1D(fils,
                     kernSize,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidDimensions))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batchSize,
              epochs=epochs,
              validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test, batch_size=batchSize)

    assert(acc > .8)


if __name__ == '__main__':
    pytest.main([__file__])
