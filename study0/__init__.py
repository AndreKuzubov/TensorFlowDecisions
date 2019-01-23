import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    pass

    # -----  1
    # 3.  # a rank 0 tensor; a scalar with shape [],
    # [1., 2., 3.]  # a rank 1 tensor; a vector with shape [3]
    # [[1., 2., 3.], [4., 5., 6.]]  # a rank 2 tensor; a matrix with shape [2, 3]
    # [[[1., 2., 3.]], [[7., 8., 9.]]]  # a rank 3 tensor with shape [2, 1, 3]

    # # ---- 2
    # a = tf.constant(3.0, dtype=tf.float32)
    # b = tf.constant(4.0)  # also tf.float32 implicitly
    # total = a + b
    # print(a)
    # print(b)
    # print(total)
    #
    # # writer = tf.summary.FileWriter('.')
    # # writer.add_graph(tf.get_default_graph())
    # sess = tf.Session()
    # # print(sess.run(total))
    # print(sess.run({'ab': (a, b), 'total': total}))

    # # ------ 3
    # vec = tf.random_uniform(shape=(3,))
    # out1 = vec + 1
    # out2 = vec + 2
    # print(sess.run(vec))
    # print(sess.run(vec))
    # print(sess.run((out1, out2)))

    # #  ------ 4  PlaceHolder
    # sess = tf.Session()
    # x = tf.placeholder(tf.float32)
    # y = tf.placeholder(tf.float32)
    # z = x + y
    #
    # print(sess.run(z, feed_dict={x: 3, y: 4.5}))
    # print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

    # # ------- 5 Datasets
    # my_data = [
    #     [0, 1, ],
    #     [2, 3, ],
    #     [4, 5, ],
    #     [6, 7, ],
    # ]
    # slices = tf.data.Dataset.from_tensor_slices(my_data)
    # next_item = slices.make_one_shot_iterator()
    # sess = tf.Session()
    # while True:
    #     try:
    #         print(sess.run(next_item.get_next()))
    #     except tf.errors.OutOfRangeError:
    #         break

    # # ------ 5.1 If the Dataset depends on stateful operations you may need to initialize the iterator before using it, as shown below
    # r = tf.random_normal([10, 3])
    # dataset = tf.data.Dataset.from_tensor_slices(r)
    # iterator = dataset.make_initializable_iterator()
    # next_row = iterator.get_next()
    #
    # sess = tf.Session()
    # sess.run(iterator.initializer)
    # while True:
    #     try:
    #         print(sess.run(next_row))
    #     except tf.errors.OutOfRangeError:
    #         break

    # #  ------- 6 Layers
    # x = tf.placeholder(tf.float32, shape=[None, 3])
    # linear_model = tf.layers.Dense(units=1)
    # y = linear_model(x)
    #
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # print(sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6],[7, 8, 9]]}))

    # #  ---- 7 Feature columns
    # features = {
    #     'sales': [[5], [10], [8], [9]],
    #     'department': ['sports', 'sports', 'gardening', 'gardening']}
    #
    # department_column = tf.feature_column.categorical_column_with_vocabulary_list('department', ['sports', 'gardening','other'])
    # department_column = tf.feature_column.indicator_column(department_column)
    #
    # columns = [
    #     tf.feature_column.numeric_column('sales'),
    #     department_column
    # ]
    # inputs = tf.feature_column.input_layer(features, columns)
    #
    # var_init = tf.global_variables_initializer()
    # table_init = tf.tables_initializer()
    # sess = tf.Session()
    # sess.run((var_init, table_init))
    # print(sess.run(inputs))

    # ------ 8 Training
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print('prediction: ', sess.run(y_pred))

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    print('loss: ', sess.run(loss))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    # print(sess.run(y_pred))
    print(sess.run(linear_model(tf.constant([[2], [2], [3], [4]], dtype=tf.float32))))

    writter = tf.summary.FileWriter('1')
    writter.add_graph(sess.graph)
    writter.close()

    pass
