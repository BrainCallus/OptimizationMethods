import math
import numpy as np
from matplotlib import pyplot as plt


def generate_descent_polynom(dev, func, dots_number, draw=False, title=False):
    k = int(math.sqrt(dev)) + dots_number / 10
    x = np.linspace(-k, k, dots_number)
    y = np.asarray([func(i) + np.random.uniform(-dev, dev) for i in x])
    y_real = np.asarray([func(i) for i in x])

    if draw:
        plt.title((title if title else "") + " $deviation = $" + str(dev))
        plt.plot(x, y, ".", label="")
        plt.show()
    return x, y, y_real