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


def loadDataSet():
    if (not os.path.exists("dataset/GTSRB_Final_Training_Images")):
        print("loading GTSRB_Final_Training_Images...")
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
        download_file("http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip",
                      local_filename="dataset/GTSRB_Final_Test_Images.zip")

        print("unzipping GTSRB_Final_Test_Images...")
        zip_ref = zipfile.ZipFile("dataset/GTSRB_Final_Test_Images.zip", 'r')
        zip_ref.extractall("dataset/GTSRB_Final_Test_Images/")
        zip_ref.close()
        os.remove("dataset/GTSRB_Final_Test_Images.zip")

        print("success GTSRB_Final_Test_Images")


def next_train_batch(batch_size=100):
    files = glob.glob("dataset/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/*/*.ppm")
    random.shuffle(files)

    x_batch = []
    y_batch = []
    for i in range(batch_size):
        f = random.choice(files)
        image = Image.open(f)
        # TODO сделать сверку без приведения к единому размеру
        image = image.resize((32, 32), Image.ANTIALIAS)
        x = np.asarray(image, dtype=float)
        x /= x.max()
        _sy = int(f[f.rfind("/") - 4:f.rfind("/")])
        y = [0] * 43
        y[_sy] = 1
        y = np.array(y)
        x_batch += [x]
        y_batch += [y]

    return x_batch, y_batch



if __name__ == "__main__":
    loadDataSet()

    x_batch, y_batch = next_train_batch()

    print()
