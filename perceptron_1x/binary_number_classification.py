import sys

# для запуска из родительской и дочерней папок
sys.path.append('../')
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
import glob
from utils import imageUtils


def generateXY(batchSize, israndom=True):
    x = np.array([[random.random() if israndom else i / float(batch_size)] for i in range(batchSize)])
    y = np.array([[0 if x[i][0] <= 0.8 else 1] for i in range(batch_size)])
    return x, y


if __name__ == "__main__":
    batch_size = 1000
    learning_rate = 1e-3
    x = tf.placeholder(tf.float32, shape=(None, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

    w = tf.Variable(tf.random_normal((1, 4)), name='w')  # weight
    b = tf.Variable(tf.random_normal((1,)), dtype=tf.float32, name='b')  # bias

    y_pred = tf.nn.sigmoid(tf.reduce_sum(tf.matmul(x, w) + b, axis=1, keep_dims=True))
    loss = tf.reduce_sum((y - y_pred) ** 2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
        .minimize(loss, var_list=[w, b])

    tf.summary.scalar(name="loss", tensor=loss)
    tf.summary.scalar(name="b", tensor=b[0])
    tf.summary.histogram(name="w", values=w)

    init = tf.initialize_all_variables()
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init)
        w_val, b_val = sess.run([w, b])
        print("init w = %s b = %.3f, start training... " % (w_val, b_val))
        print()

        writter = tf.summary.FileWriter('log/binary_number_classification/tmp')
        writter.add_graph(sess.graph)

        for i in range(10000):
            x_batch, y_batch = generateXY(batchSize=batch_size)

            summary, _, loss_val, w_val, b_val, x_val, y_val = sess.run(
                [merged_summary_op, optimizer, loss, w, b, x, y],
                feed_dict={x: x_batch, y: y_batch})
            writter.add_summary(summary, i)
            if (i % 200 == 0):
                x_test_batch, y_test_batch = generateXY(batchSize=batch_size, israndom=False)
                y_pred_val, = sess.run([y_pred], feed_dict={x: x_test_batch})

                plt.plot(x_test_batch, y_pred_val, x_test_batch, y_test_batch)
                plt.savefig('log/binary_number_classification/classifications_%s.png' % (i))
                # plt.show()
                plt.close()

                print("i = %s loss = %.8f w = %s b = %.3f learning_rate = %.4f" % (
                    i, loss_val, w_val, b_val, learning_rate))
                print()

                if (loss_val < 1):
                    break
        imageUtils.createGif(
            imageFileNames=sorted(glob.glob('log/binary_number_classification/classifications_*.png'),
                                  key=lambda a: int(a[a.rfind('_') + 1:a.rfind('.')])),
            saveFileName='log/binary_number_classification/classifications.gif'
        )
        writter.close()
