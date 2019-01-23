import tensorflow as tf
from tensorflow import keras
import numpy as np


class MyLayer(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Be sure to call this at the end
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    pass

    # #  ----- 1 Sequential model multi-layer perceptron
    # model = keras.Sequential()
    # # Adds a densely-connected layer with 64 units to the model:
    # model.add(keras.layers.Dense(64, activation='relu'))
    # # Add another:
    # model.add(keras.layers.Dense(64, activation='relu'))
    # # Add a softmax layer with 10 output units:
    # model.add(keras.layers.Dense(10, activation='softmax'))
    #
    # # train = tf.train.*
    # model.compile(optimizer=tf.train.AdamOptimizer(0.001),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    # #  ----2 configure layers
    # # Create a sigmoid layer:
    # tf.layers.Dense(64, activation='sigmoid')
    # # Or:
    # tf.layers.Dense(64, activation=tf.sigmoid)
    #
    # # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
    # tf.layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
    # # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
    # tf.layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))
    #
    # # A linear layer with a kernel initialized to a random orthogonal matrix:
    # tf.layers.Dense(64, kernel_initializer='orthogonal')
    # # A linear layer with a bias vector initialized to 2.0s:
    # tf.layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    # Create a model using the custom layer
    model = keras.Sequential(
        [
            tf.keras.layers.Dense(10),
            keras.layers.Activation('softmax')
        ])

    # The compile step specifies the training configuration
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains for 5 epochs.
    model.fit(data, labels, batch_size=32, epochs=5)

    pass
