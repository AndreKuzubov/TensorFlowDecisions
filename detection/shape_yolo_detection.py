import sys

# для запуска из родительской и дочерней папок
sys.path.append('../')
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from utils import imageUtils

if __name__ =="__main__":
    pass
    # TODO from https://towardsdatascience.com/object-detection-with-neural-networks-a4e2c46b4491
    # Create images with random rectangles and bounding boxes.
    # num_imgs = 50000
    #
    # img_size = 8
    # min_object_size = 1
    # max_object_size = 4
    # num_objects = 1
    #
    # bboxes = np.zeros((num_imgs, num_objects, 4))
    # imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0
    #
    # for i_img in range(num_imgs):
    #     for i_object in range(num_objects):
    #         w, h = np.random.randint(min_object_size, max_object_size, size=2)
    #         x = np.random.randint(0, img_size - w)
    #         y = np.random.randint(0, img_size - h)
    #         imgs[i_img, x:x + w, y:y + h] = 1.  # set rectangle to 1
    #         bboxes[i_img, i_object] = [x, y, w, h]
    #
    # i = 0
    # plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    # for bbox in bboxes[i]:
    #     plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
    # plt.show()