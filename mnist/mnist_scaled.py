import sys
import os

#
# для запуска из родительской и дочерней папок
sys.path.append('../')
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist_data
import random
# import matplotlib.pyplot as plt
import numpy as np
# import glob
# from utils import imageUtils
from utils.specificFixs import *
from PIL import Image
# import PIL.ImageColor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

MODEL_PATH = "log/mnist_scaled/model.hdf5"
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

mnist = mnist_data.input_data.read_data_sets("log/", one_hot=True)

INPUT_IMAGE_SIZE = 256


def showImages(imagesX, labelsY):
    imagesX = imagesX.reshape((-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    imagesX *= 255.
    for i, image2d in enumerate(imagesX):
        image = Image.fromarray(image2d.astype(np.uint8))
        image.show(title=str(list(labelsY[i]).index(1)))


def next_batch(batch_size=64, istestbatch=False):
    print('generate train/test batch %s ' % (batch_size,))
    batch_xs, batch_ys = mnist.train.next_batch(batch_size) if not istestbatch else mnist.test.next_batch(batch_size)

    batch_size = len(batch_xs)
    batch_xs = batch_xs.reshape([-1, 28, 28, 1])

    batch_scaled_xs = np.zeros([batch_size, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1])
    batch_offseted_xs = np.zeros([batch_size, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1])

    for batch_index in range(batch_size):
        scale = max(random.random(), 0.2) * INPUT_IMAGE_SIZE / 28
        for i in range(INPUT_IMAGE_SIZE):
            if (i / scale > 28):
                break
            for j in range(INPUT_IMAGE_SIZE):
                if (j / scale > 28):
                    break
                batch_scaled_xs[batch_index][i][j] = batch_xs[batch_index][int(i / scale)][int(j / scale)]

        offset_i = int(random.random() * (INPUT_IMAGE_SIZE - 28 * scale))
        offset_j = int(random.random() * (INPUT_IMAGE_SIZE - 28 * scale))
        for i in range(INPUT_IMAGE_SIZE - offset_i - 1):
            for j in range(INPUT_IMAGE_SIZE - offset_j - 1):
                batch_offseted_xs[batch_index][offset_i + i][offset_j + j] = batch_scaled_xs[batch_index][i][j]

    return batch_offseted_xs, batch_ys


def training():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid'))

    model.add(Convolution2D(32, kernel_size=(5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid'))

    model.add(Convolution2D(64, kernel_size=(5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    for i in range(3):
        batch_xs, batch_ys = next_batch(batch_size=2000)
        test_xs, test_ys = next_batch(batch_size=100, istestbatch=True)

        model.fit(batch_xs, batch_ys,
                  verbose=1,
                  epochs=10,
                  validation_data=(test_xs, test_ys),
                  validation_split=0.1,
                  callbacks=[ModelCheckpoint(filepath=MODEL_PATH,
                                             monitor="val_acc",
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode="auto")
                             ]
                  )
    print("finish...")



if __name__ == "__main__":
    # print('mnist train size: %s ' % (mnist.train.num_examples))
    # training()

    model = load_model(MODEL_PATH)

    test_xs, test_ys = next_batch(batch_size=1000, istestbatch=True)
    score = model.evaluate(test_xs, test_ys, verbose=0)
    print("Test score %f " % score[0])
    print("Test accuracy %f " % score[1])
