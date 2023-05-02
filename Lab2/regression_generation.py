import math
import numpy as np
from matplotlib import pyplot as plt

def generate_descent(dev, deg=1, x_int=0, y_int=0, draw=False):
    dens = 20
    smt = 3
    if dev == 0 and deg != 1:
        k = int(math.log(dens, deg))
    elif deg == 1 and dev != 0:
        k = dev + smt
    elif deg == 1 and dev == 0:
        k = smt
    else:
        k = int(math.log(dev * dens, deg))

    x = np.linspace(-k, k, dens * k)
    y = [(i + x_int) ** deg + y_int + np.random.uniform(-dev, dev) for i in x]

    if draw:
        a = {
            x_int > 0: "(x + " + str(x_int) + ")",
            x_int < 0: "(x - " + str(- x_int) + ")",
            x_int == 0: "x"
        }[True]
        b = {
            deg != 1: "^{" + str(deg) + "}",
            deg == 1: ""
        }[True]
        c = {
            y_int > 0: " + " + str(y_int),
            y_int < 0: " - " + str(- y_int),
            y_int == 0: ""
        }[True]

        plt.title("$f(x) = " + a + b + c + "; \quad deviation = "
                  + str(dev) + "$")
        plt.plot(x, y, ".", label="")
        plt.show()
    return x, y

def generate_descent_polynom(dev, func, draw=False, title=False):
    dens = 30
    k = int(math.sqrt(dev)) + 12
    x = np.linspace(-k, k, dens * k)
    y = [func(i) + np.random.uniform(-dev, dev) for i in x]

    if draw:
        plt.title((title if title else "") + " $deviation = $" + str(dev))
        plt.plot(x, y, ".", label="")
        plt.show()
    return x, y