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
from utils.inetUtils import *
import time
import requests
import os
import zipfile
import glob
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

MODEL_PATH = "log/road_signs_scaled/road_signs_scaled.hdf5"
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

INPUT_IMAGE_SIZE = 256


def showImages(imagesX, batchY):
    imagesX = imagesX.reshape((-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))
    imagesX *= 255.
    for i, image2d in enumerate(imagesX):
        image = Image.fromarray(image2d.astype(np.uint8))
        image.show(title=str(i) + str(list(batchY[i]).index(1)))


def loadDataSet():
    if (not os.path.exists("dataset/GTSRB_Final_Training_Images")):
        print("loading GTSRB_Final_Training_Images...")
        os.makedirs("dataset/GTSRB_Final_Training_Images")
        download_file("http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip",
                      local_filename="dataset/GTSRB_Final_Training_Images.zip")

        print("unzipping GTSRB_Final_Training_Images...")
        zip_ref = zipfile.ZipFile("dataset/GTSRB_Final_Training_Images.zip", 'r')
        zip_ref.extractall("dataset/GTSRB_Final_Training_Images/")
        zip_ref.close()
        os.remove("dataset/GTSRB_Final_Training_Images.zip")

        print("success GTSRB_Final_Training_Images")

    if (not os.path.exists("dataset/GTSRB_Final_Test_Images")):
        print("loading GTSRB_Final_Test_Images...")
        os.makedirs("dataset/GTSRB_Final_Test_Images")
        download_file("http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip",
                      local_filename="dataset/GTSRB_Final_Test_Images.zip")

        print("unzipping GTSRB_Final_Test_Images...")
        zip_ref = zipfile.ZipFile("dataset/GTSRB_Final_Test_Images.zip", 'r')
        zip_ref.extractall("dataset/GTSRB_Final_Test_Images/")
        zip_ref.close()
        os.remove("dataset/GTSRB_Final_Test_Images.zip")

        print("success GTSRB_Final_Test_Images")


def next_batch(batch_size=100):
    files = glob.glob("dataset/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/*/*.ppm")
    random.shuffle(files)

    x_batch = []
    y_batch = []
    for i in range(batch_size):
        f = random.choice(files)
        base_img = Image.new('RGB', (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), color='black')

        image = Image.open(f)
        sign_size = int(max(random.random() * INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE * 0.2))
        image = image.resize((sign_size, sign_size), Image.ANTIALIAS)
        paste_point = (
            int(random.random() * (INPUT_IMAGE_SIZE - sign_size)),
            int(random.random() * (INPUT_IMAGE_SIZE - sign_size)))
        base_img.paste(image, box=(paste_point))

        x = np.asarray(base_img, dtype=float)
        x /= x.max()

        _sy = int(f[f.rfind("/") - 4:f.rfind("/")])
        y = [0] * 43
        y[_sy] = 1
        y = np.array(y)

        x_batch += [x]
        y_batch += [y]

    return np.array(x_batch), np.array(y_batch)


def train():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

    model.add(Convolution2D(32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Convolution2D(32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Convolution2D(64, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Convolution2D(64, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    for i in range(3):
        print(" i = %s" % (i,))
        batch_xs, batch_ys = next_batch(batch_size=2000)
        test_xs, test_ys = next_batch(batch_size=100)

        model.fit(batch_xs, batch_ys,
                  verbose=1,
                  epochs=10,
                  validation_data=(test_xs, test_ys),
                  validation_split=0.1,
                  callbacks=[ModelCheckpoint(filepath=MODEL_PATH,
                                             monitor="val_acc",
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode="auto")
                             ]
                  )
        del batch_xs
        del batch_ys
        del test_xs
        del test_ys
    print("finish...")


if __name__ == "__main__":
    loadDataSet()

    train()

    model = load_model(MODEL_PATH)
    test_xs, test_ys = next_batch(batch_size=1000)
    score = model.evaluate(test_xs, test_ys, verbose=0)
    print("Test score %f " % score[0])
    print("Test accuracy %f " % score[1])
