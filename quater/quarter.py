import numpy as np
import tensorflow as tf
import random
import json
import os
from tensorflow import keras

if __name__ == '__main__':
    pass

    lx = [[random.random(), random.random()] for i in range(0, 100)]
    ly = [[
        float(1 if ilx[0] >= 0 and ilx[1] >= 0 else 0),
        float(1 if ilx[0] < 0 and ilx[1] >= 0 else 0),
        float(1 if ilx[0] < 0 and ilx[1] < 0 else 0),
        float(1 if ilx[0] >= 0 and ilx[1] < 0 else 0)
    ] for ilx in lx]

    x = np.array(lx)
    y_true = np.array(ly)

    if os.path.exists('quarter_model.ckpt'):
        model = keras.models.load_model('quarter_model.ckpt')
    else:
        model = tf.keras.models.Sequential([
            keras.layers.Dense(units=4, input_shape=(2,))
        ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=tf.losses.absolute_difference)

    # для восстановленной модели
    print('model loss = ', np.mean(model.predict(x) - y_true))

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint("quarter_model.ckpt")

    model.fit(x, y_true, epochs=10,callbacks=[cp_callback])

    print('predict: ', x[10], model.predict(np.array([x[10]])))
    # model.save('quarter_model.ckpt')
    with open('quater_model_config.json', 'w+') as file:
        file.write(model.to_json())
        file.close()

    print('model trained: loss = ', np.mean(model.predict(x) - y_true))
