import sys
import os

# для запуска из родительской и дочерней папок
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from utils import imageUtils
from mpl_toolkits.mplot3d import Axes3D
import glob


def plot3d_rotate_gif(zfunction, saveAsFile, X=np.arange(-10, 10, 0.01), Y=np.arange(-10, 10, 0.01)):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(X, Y)
    Z = zfunction(X, Y)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=10)

    for angle in range(0, 360, 5):
        print("generate %s angle = %s" % (saveAsFile, angle))
        ax.view_init(30, angle)
        plt.savefig(saveAsFile + str(angle) + '.png')

    imageUtils.createGif(
        imageFileNames=sorted(glob.glob(pathname=saveAsFile + "*.png"),
                              key=lambda a: int(a[len(saveAsFile):-len('.png')])),
        saveFileName=saveAsFile,
        frameDuraction=0.05
    )

    for f in glob.glob(pathname=saveAsFile + "*.png"):
        os.remove(f)


squaredDiff = lambda X, Y: (X - Y) ** 2

squaredEntropy = lambda X, Y: (-X * np.log(Y))

absDiff = lambda X, Y: abs(X - Y)

if __name__ == "__main__":
    plot3d_rotate_gif(squaredDiff, saveAsFile='log/squaredDiff.gif')
    plot3d_rotate_gif(absDiff, saveAsFile='log/absDiff.gif')
    plot3d_rotate_gif(squaredEntropy, saveAsFile='log/squaredEntropy.gif', Y=np.arange(0.001, 1, 0.01), X=np.arange(-1, 1, 0.01))
