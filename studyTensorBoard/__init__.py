import tensorflow as tf

if __name__ == "__main__":
    with tf.name_scope('hidden') as scope:
        a = tf.constant(5, name='alpha')
        W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
        b = tf.Variable(tf.zeros([1]), name='biases')

    sess = tf.Session()
    writter = tf.summary.FileWriter('1')
    writter.add_graph(sess.graph)
    writter.close()