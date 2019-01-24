import sys

# для запуска из родительской и дочерней папок
sys.path.append('../')
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist_data
import random
import matplotlib.pyplot as plt
import numpy as np
import glob
from utils import imageUtils
from utils.specificFixs import *
from PIL import Image
import PIL.ImageColor

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

def showImages(batchImages1d):
    images2d = np.reshape(batchImages1d, (-1,28,28))
    images2d *=255.
    for image2d in images2d:
        image = Image.fromarray(image2d.astype(np.uint8))
        image.show()




if __name__ == "__main__":
    mnist = mnist_data.input_data.read_data_sets("log/",one_hot=True)

    # show images
    imagesBatch,_ = mnist.train.next_batch(10)
    showImages(imagesBatch)

    x = tf.placeholder(tf.float32, [None, 784])

    W_relu = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
    b_relu = tf.Variable(tf.truncated_normal([100]))
    h = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)

    keep_probability = tf.placeholder(tf.float32)
    h_drop = tf.nn.dropout(h, keep_probability)

    W = tf.Variable(tf.zeros([100, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h_drop, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])

    # функция потерь
    logit = tf.matmul(h_drop, W) + b
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y_))

    # обучение
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(4000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_probability: 0.5})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Точность: %s" %
          sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_probability: 1.}))
