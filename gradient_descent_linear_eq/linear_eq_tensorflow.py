import random
import tensorflow as tf


def generateXY(batch_size):
    k = 10 + random.random() * 2 - 1
    l = 1 + random.random() * 2 - 1
    x = [[random.random()] for i in range(batch_size)]
    y = [[k * x_item[0] + l] for x_item in x]

    return x, y


if __name__ == "__main__":
    h = 1e-5

    batch_size = 1000

    x = tf.placeholder(tf.float32, shape=(batch_size, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(batch_size, 1,), name='y')
    k = tf.Variable(tf.random_normal((1,), dtype=tf.float32), name='k')
    l = tf.Variable(tf.random_normal((1,), dtype=tf.float32), name='l')

    y_pred = tf.mul(x, k) + l
    loss = tf.reduce_sum((y - y_pred) ** 2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=h).minimize(loss)

    tf.summary.scalar(name="loss", tensor=loss)
    tf.summary.scalar(name="k", tensor=k[0])
    tf.summary.scalar(name="l", tensor=l[0])

    init = tf.initialize_all_variables()
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:

        sess.run(init)
        k_val, l_val = sess.run([k, l])
        print('random init k = %.03f' % (k_val))
        print('random init l = %.03f' % (l_val))

        writter = tf.summary.FileWriter('log/linear_eq_tensorflow/tmp')
        writter.add_graph(sess.graph)

        for i in range(0, 6000):
            x_data, y_data = generateXY(batch_size)

            summary, _, loss_val, k_val, l_val = sess.run(
                [merged_summary_op, optimizer, loss, k, l],
                feed_dict={x: x_data, y: y_data})

            writter.add_summary(summary, i)

        writter.close()
