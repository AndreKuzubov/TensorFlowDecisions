import sys

#
# для запуска из родительской и дочерней папок
sys.path.append('../')
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist_data
# import random
# import matplotlib.pyplot as plt
import numpy as np
# import glob
# from utils import imageUtils
from utils.specificFixs import *
from PIL import Image
# import PIL.ImageColor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

MODEL_PATH = "log/mnist_scale_stability/model.hdf5"
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

mnist = mnist_data.input_data.read_data_sets("log/", one_hot=True)


def next_batch(batch_size=64):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([-1, 28, 28, 1])
    return batch_xs, batch_ys


def next_test_batch():
    batch_xs, batch_ys = mnist.test.images, mnist.test.labels
    batch_xs = batch_xs.reshape([-1, 28, 28, 1])
    return batch_xs, batch_ys


def showImages(batchImages1d):
    images2d = np.reshape(batchImages1d, (-1, 28, 28))
    images2d *= 255.
    for image2d in images2d:
        image = Image.fromarray(image2d.astype(np.uint8))
        image.show()


def training():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Convolution2D(64, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    batch_xs, batch_ys = next_batch(batch_size=10000)
    test_xs, test_ys = next_test_batch()

    model.fit(batch_xs, batch_ys,
              verbose=1,
              epochs=10,
              validation_data=(test_xs, test_ys),
              validation_split=0.1,
              callbacks=[ModelCheckpoint(filepath=MODEL_PATH,
                                         monitor="val_acc",
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode="auto")
                         ]
              )
    print("finish...")

    score = model.evaluate(test_xs, test_ys, verbose=0)
    print("Test score %f " % score[0])
    print("Test accuracy %f " % score[1])


def test_resable_images():
    model = load_model(MODEL_PATH)

    pass


if __name__ == "__main__":
    print('mnist train size: %s ' % (mnist.train.num_examples))
    training()
