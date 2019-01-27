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
from utils.tensorUtils import *
from PIL import Image
import PIL.ImageColor

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


def showImages(batchImages1d):
    images2d = np.reshape(batchImages1d, (-1, 28, 28))
    images2d *= 255.
    for image2d in images2d:
        image = Image.fromarray(image2d.astype(np.uint8))
        image.show()


if __name__ == "__main__":
    mnist = mnist_data.input_data.read_data_sets("log/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # x layer
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # layer 1
    W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b_conv_1 = tf.Variable(tf.truncated_normal([32], stddev=0.1))
    conv_1 = tf.nn.conv2d(x_image, W_conv_1, strides=[1, 1, 1, 1], padding="SAME") + b_conv_1
    h_conv_1 = tf.nn.relu(conv_1)
    h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # layer 2
    W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv_2 = tf.Variable(tf.truncated_normal([64], stddev=0.1))
    conv_2 = tf.nn.conv2d(h_pool_1, W_conv_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv_2
    h_conv_2 = tf.nn.relu(conv_2)
    h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # layer 3 - perceptron
    h_pool_flat = tf.reshape(h_pool_2, [-1, 7 * 7 * 64])
    W_fc_1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
    b_fc_1 = tf.Variable(tf.truncated_normal([1024], stddev=0.1))
    h_fc_1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc_1) + b_fc_1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob=keep_prob)

    # layer 4 - perceptron
    W_fc_2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b_fc_2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

    # y layer
    logic_conv = tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2
    y_conv = tf.nn.softmax(logic_conv)

    # error
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logic_conv, labels=y))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    corrent_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

    loc = locals()
    initAllSummaries(
        {k: loc[k] for k in loc.keys() if isinstance(loc[k], (tf.Variable, tf.Tensor))}
    )
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        writter = tf.summary.FileWriter('log/mnist_conv/tmp')
        writter.add_graph(sess.graph)
        for i in range(100):
            batch_xs, batch_ys = mnist.train.next_batch(64)

            _, accuracy_val, summary = sess.run((train_step, accuracy, merged_summary_op,),
                                                feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            writter.add_summary(summary)
            print("train accuracy = %s" % (accuracy_val,))

        writter.close()
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.}))
