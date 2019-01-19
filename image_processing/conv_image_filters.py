import sys

# для запуска из родительской и дочерней папок
from numba.testing.ddt import feed_data

sys.path.append('../')
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import glob
from utils import imageUtils
from PIL import Image
from utils.specificFixs import *
import time


def boxFilter(sourceImage, boxSize=3, boxScalar=None, padding="VALID"):
    """
    box фильтр - размытие изображения
    :param sourceImage: исходное изображение
    :param boxScalar: множетель ядра фильтра - определяет яркость полученного зображения
            если boxScalar = 1/boxSize**2 - яркость не меняется
    :param padding: исходное изображение
    :return: обработанное изображение
    """
    if (boxScalar is None):
        boxScalar = 1. / boxSize ** 2

    x_image = tf.constant(np.asarray(sourceImage).astype(np.float32), dtype=tf.float32)
    x_image = tf.transpose(x_image, [2, 0, 1])
    x_image = tf.reshape(x_image, list(x_image.get_shape()) + [1])


    kernel = tf.constant(boxScalar * np.asarray([[1] * boxSize] * boxSize), dtype=tf.float32)
    kernel = tf.reshape(kernel, [boxSize, boxSize, 1, 1])

    filtered = tf.nn.conv2d(x_image, kernel, strides=[1, 1, 1, 1], padding=padding)

    with tf.Session() as sess:
        y_image, = sess.run([filtered])

    y_image = y_image.transpose((3, 1, 2, 0,))
    y_image = y_image.reshape( y_image.shape[1:])

    # обработка засветов
    y_image = np.minimum(y_image,255)
    y_image = np.maximum(y_image,0)
    newImage = Image.fromarray(y_image.astype(np.uint8), 'RGB')
    return newImage

def wbImage(sourceImage):
    """
    чб фильтр - делает изображение черно-белым
    :param sourceImage: исходное изображение
    :return: обработанное изображение
    """
    x = tf.constant(np.asarray(sourceImage), dtype=tf.float32)

    if (x.shape[2] == 4):
        y = tf.constant(np.asarray(
            [
                [1 / 3, 1 / 3, 1 / 3, 0],
                [1 / 3, 1 / 3, 1 / 3, 0],
                [1 / 3, 1 / 3, 1 / 3, 0],
                [0, 0, 0, 1],
            ]
        ), dtype=tf.float32)
    else:
        y = tf.constant(np.asarray(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
            ]
        ), dtype=tf.float32)

    out = tf.tensordot(x, y, axes=[[2], [0]])
    with tf.Session() as sess:
        result = sess.run(out)
        sess.close()
    return Image.fromarray(np.array(result, dtype=np.uint8))


if __name__ == "__main__":
    start_time = time.time()

    source = Image.open("log/source_image.jpg")

    box_filtered = boxFilter(sourceImage=source, boxSize=10)
    box_filtered.show()
    box_filtered.save("log/box_filter.jpg")

    box_filtered = boxFilter(sourceImage=source, boxSize=3,boxScalar=1./(9.*2.))
    box_filtered.show()
    box_filtered.save("log/box_filter_dark.jpg")

    box_filtered = boxFilter(sourceImage=source, boxSize=3, boxScalar=2. / (9.))
    box_filtered.show()
    box_filtered.save("log/box_filter_light.jpg")

    box_filtered = boxFilter(sourceImage=source, boxSize=20, padding="SAME")
    box_filtered.show()
    box_filtered.save("log/box_filter_same.jpg")

    box_filtered = wbImage(source)
    box_filtered.show()
    box_filtered.save("log/wbImage.jpg")

    print("finished: %s seconds" % (time.time() - start_time))