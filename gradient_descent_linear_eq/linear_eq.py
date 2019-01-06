import random
import matplotlib.pyplot as plt
import numpy as np


def generateXY():
    k = 10 + random.random() * 2 - 1
    l = 1 + random.random() * 2 - 1
    x = random.random()

    y = k * x + l
    return x, y


if __name__ == "__main__":

    k = random.random()
    l = random.random()
    h = 0.001
    losses = []
    ks = []
    ls = []
    while True:
        x, y = generateXY()

        loss = abs(y - (k * x + l))
        losses += [loss]
        ls += [l]
        ks += [k]
        k_next = k - 2 * h * x * (k * x + l - y)
        l_next = l - 2 * h * (k * x + l - y)

        k = k_next
        l = l_next

        if (len(losses) > 10000 and np.std(losses[-10000:]) < 0.37):
            print("break")
            k = np.mean(ks[-10000:])
            l = np.mean(ls[-10000:])
            break

    print("len: " + str(len(ls)) +
          "k = " + str(k) + " l = " + str(l))
    plt.plot(losses)
    plt.savefig('log/linear_eq/loss_scalars.png')
    plt.show(dpi=2000, figsize=(200, 200))

    plt.plot(
        range(len(ks)),
        ks,
        range(len(ls)),
        ls,
    )
    plt.savefig('log/linear_eq/k_l_scalars.png')
    plt.show(dpi=20000, figsize=(2000, 2000))
