import math
import numpy as np
from matplotlib import pyplot as plt


def generate_descent_polynom(dev, func, dens=10, dots_number=None, draw=False, title=False):
    k = int(math.sqrt(dev)) + dens // 2
    if dots_number is None:
        dots_number = dens * k
    x = np.linspace(-k, k, dots_number)
    y = [func(i) + np.random.uniform(-dev, dev) for i in x]

    if draw:
        plt.title((title if title else "") + " $deviation = $" + str(dev))
        plt.plot(x, y, ".", label="")
        plt.show()
    return x, y