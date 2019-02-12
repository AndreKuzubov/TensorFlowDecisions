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
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

MODEL_PATH = "log/road_signs_scaled/road_signs_scaled.hdf5"
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

INPUT_IMAGE_SIZE = 256

IMAGE_CELL_COUNT = 30
YOLO_SMALL_WINDOW_SIZE_BY_CELLS = 10
YOLO_WINDOW_SIZE_STEP = 5

YOLO_TEST_PATH = "log/road_signs_scaled/yolotest/%s__%s_%s.png"
YOLO_TEST_SOURCE_PATH = "log/road_signs_scaled/yolotest/source.png"
if not os.path.exists(os.path.dirname(YOLO_TEST_SOURCE_PATH)):
    os.makedirs(os.path.dirname(YOLO_TEST_SOURCE_PATH))
COLORS = [
    "#d68a59",
    "#6a5f31",
    "#f0d698",
    "#f4c430",
    "#424632",
    "#bd33a4",
    "#48d1cc",
    "#ff7514",
    "#d76e00",
    "#ceff1d",
]

def showImages(imagesX, batchY=None, saveAsFile=None):
    imagesX = imagesX.copy()
    imagesX = imagesX.reshape((-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))
    imagesX *= 255.


    for i, image2d in enumerate(imagesX):
        image = Image.fromarray(image2d.astype(np.uint8))
        if (saveAsFile is None):
            image.show(title=str(i))
        else:
            image.save(saveAsFile)


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


def predinctDetect(model, batchX):
    predics = []
    for batchIndex, imageX in enumerate(batchX):
        imageX = imageX.reshape([1, imageX.shape[0], imageX.shape[1], imageX.shape[2]])
        # showImages(imageX, saveAsFile=YOLO_TEST_SOURCE_PATH)
        showImages(imageX)


        imageWidhtPx = imageX.shape[1]
        imageHeightPx = imageX.shape[2]
        cellSizePx = min(imageWidhtPx, imageHeightPx) / IMAGE_CELL_COUNT

        cellsCountHorizontal = int(imageWidhtPx / cellSizePx)
        cellsCountVertical = int(imageHeightPx / cellSizePx)

        cellsPredictCl = np.zeros((cellsCountHorizontal, cellsCountVertical, 43))

        cell_participation_count = 0
        for windowSizeCl in range(YOLO_SMALL_WINDOW_SIZE_BY_CELLS, IMAGE_CELL_COUNT, YOLO_WINDOW_SIZE_STEP):
            print("windowSizeCl = %s" % (windowSizeCl))
            cell_participation_count += windowSizeCl
            for cell_sx in range(0, cellsCountHorizontal - windowSizeCl, 1):
                for cell_sy in range(0, cellsCountVertical - windowSizeCl, 1):

                    horizontalCropping = (
                        int(cell_sx * cellSizePx), int(imageWidhtPx - (cell_sx + windowSizeCl) * cellSizePx))
                    verticalCropping = (
                        int(cell_sy * cellSizePx), int(imageHeightPx - (cell_sy + windowSizeCl) * cellSizePx))

                    preprocessModel = Sequential()
                    preprocessModel.add(Cropping2D(cropping=(verticalCropping, horizontalCropping)))
                    preprocessModel.add(
                        Lambda(lambda image: tf.image.resize_images(image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))))
                    windowImage = preprocessModel.predict(imageX)

                    predictY = model.predict(windowImage)

                    for i in (cell_sx, cell_sx + windowSizeCl):
                        for j in (cell_sy, cell_sy + windowSizeCl):
                            for class_index in range(43):
                                cellsPredictCl[i][j][class_index] += predictY[0][class_index]
                    # testing
                    # showImages(windowImage, saveAsFile=YOLO_TEST_PATH % (windowSizeCl, cell_sx, cell_sy))
        # TODO test cell_participation_count
        predics += [cellsPredictCl / cell_participation_count]

    return np.array(predics)


def road_sign_yolo_detect():
    model = load_model(MODEL_PATH)
    test_xs, test_ys = next_batch(batch_size=1)

    predict_y = predinctDetect(model, test_xs)
    showImages(test_xs,predict_y)


if __name__ == "__main__":
    loadDataSet()

    road_sign_yolo_detect()
    print()
