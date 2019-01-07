import sys

# для запуска из родительской и дочерней папок
sys.path.append('../')
import numpy as np
import tensorflow as tf
import random
import json
import os
from tensorflow import keras
from utils import imageUtils
from itertools import product
# import if needed
from utils.specificFixs import *


def generateXY(x=np.arange(-1, 1, 0.1), y=np.arange(-1, 1, 0.1), israndom=True):
    # points
    lx = np.array(list(product(x, y)))
    if (israndom):
        lx = np.array([[random.choice(x), random.choice(y)] for _ in lx])
    # a quarter of a coordinate axis
    ly = np.array([[
        float(1 if ilx[0] >= 0 and ilx[1] >= 0 else 0),
        float(1 if ilx[0] < 0 and ilx[1] >= 0 else 0),
        float(1 if ilx[0] < 0 and ilx[1] < 0 else 0),
        float(1 if ilx[0] >= 0 and ilx[1] < 0 else 0)
    ] for ilx in lx])
    return lx, ly


def getModel():
    if os.path.exists('quarter_model.ckpt'):
        model = keras.models.load_model('quarter_model.ckpt')
    else:
        model = tf.keras.models.Sequential(
            # layers
            [keras.layers.Dense(units=4, input_dim=2,name="first_layer_dense")]
        )
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.losses.mean_squared_error)
    return model


if __name__ == '__main__':
    x_batch, y_batch = generateXY()
    model = getModel()

    print('model loss = %.3f' % (np.mean((model.predict(x_batch) - y_batch)**2)))
    tb_callback = keras.callbacks.TensorBoard(log_dir='./log/four_quarters', histogram_freq=0, write_graph=True, write_images=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint("quarter_model.ckpt")
    model.fit(x_batch, y_batch, epochs=100, callbacks=[cp_callback, tb_callback])
    model.save('quarter_model.ckpt')

    x_batch, y_batch = generateXY()
    print('model trained = %.3f' % (np.mean((model.predict(x_batch) - y_batch)**2)))
