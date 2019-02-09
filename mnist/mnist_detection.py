import sys

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

MODEL_PATH = "log/mnist_detection/model.hdf5"
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

mnist = mnist_data.input_data.read_data_sets("log/", one_hot=True)

IMAGE_SIZE = 256
DETECTOR_BOX_SIZES = 10


def showImages(imagesX, imageY):
    imagesX *= 255.

    shapeImages = imagesX.shape
    images_rgb = np.zeros([shapeImages[0], shapeImages[1], shapeImages[2], 3])
    for batchIndex in range(shapeImages[0]):
        for i in range(shapeImages[1]):
            for j in range(shapeImages[2]):
                images_rgb[batchIndex][i][j] = np.array([imagesX[batchIndex][i][j][0]] * 3)

        for box_i in range(DETECTOR_BOX_SIZES):
            for box_j in range(DETECTOR_BOX_SIZES):
                if (max(imageY[batchIndex][box_i][box_j]) > 0):
                    for i in range(int(box_i * IMAGE_SIZE / DETECTOR_BOX_SIZES),
                                   int((box_i + 1) * IMAGE_SIZE / DETECTOR_BOX_SIZES)):
                        for j in range(int(box_j * IMAGE_SIZE / DETECTOR_BOX_SIZES),
                                       int((box_j + 1) * IMAGE_SIZE / DETECTOR_BOX_SIZES)):
                            images_rgb[batchIndex][i][j][2] += 200


    for image2d in images_rgb:
        image = Image.fromarray(image2d.astype(np.uint8), mode="RGB")
        image.show()


def next_batch(batch_size=64, istestbatch=False):
    """

    :param batch_size: размер пакета
    :return:
    """
    # TODO background noise
    # TODO object count
    batch_xs, batch_ys = mnist.train.next_batch(batch_size) if not istestbatch else mnist.test.next_batch(batch_size)

    batch_size = len(batch_xs)
    batch_xs = batch_xs.reshape([-1, 28, 28, 1])

    batch_large_xs = np.zeros([batch_size, IMAGE_SIZE, IMAGE_SIZE, 1])
    batch_large_ys = np.zeros([batch_size, DETECTOR_BOX_SIZES, DETECTOR_BOX_SIZES, 10])

    for batch_index in range(batch_size):
        offset_i = int(random.random() * (IMAGE_SIZE - 28))
        offset_j = int(random.random() * (IMAGE_SIZE - 28))
        for i in range(28):
            for j in range(28):
                batch_large_xs[batch_index][offset_i + i][offset_j + j] = batch_xs[batch_index][i][j]

        box_start_i = int(offset_i * DETECTOR_BOX_SIZES / IMAGE_SIZE)
        box_end_i = int((offset_i + 28) * DETECTOR_BOX_SIZES / IMAGE_SIZE)
        box_start_j = int((offset_j) * DETECTOR_BOX_SIZES / IMAGE_SIZE)
        box_end_j = int((offset_j + 28) * DETECTOR_BOX_SIZES / IMAGE_SIZE)
        for i in range(box_start_i, box_end_i + 1):
            for j in range(box_start_j, box_end_j + 1):
                batch_large_ys[batch_index][i][j] = batch_ys[batch_index]

    return batch_large_xs, batch_large_ys


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
    model.add(Dense(DETECTOR_BOX_SIZES * DETECTOR_BOX_SIZES * 10))
    model.add(Activation('softmax'))
    model.add(Reshape((DETECTOR_BOX_SIZES, DETECTOR_BOX_SIZES, 10)))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    batch_xs, batch_ys = next_batch(batch_size=1000)
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

    score = model.evaluate(test_xs, test_ys, verbose=0)
    print("Test score %f " % score[0])
    print("Test accuracy %f " % score[1])


if __name__ == "__main__":
    print('mnist train size: %s ' % (mnist.train.num_examples))
    training()
    model = load_model(MODEL_PATH)
