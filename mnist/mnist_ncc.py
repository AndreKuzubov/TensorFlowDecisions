import os, sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.examples.tutorials.mnist import input_data
import numpy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import utils

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    batch_size, img_rows, img_cols = 64, 28, 28
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    Y_train = utils.to_categorical(y_train, 10)
    Y_test = utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Convolution2D(32, 5, 5, padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Convolution2D(64, 5, 5, padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Test score: %f" % score[0])
    print("Test accuracy: %f" % score[1])
