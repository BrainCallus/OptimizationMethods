import logging
from gauss_newton import *
from powell_dogLeg import *
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

NOISE = 10
init_coefs = [-0.033, 0.4, -0.7, -0.00012, 145]


def funcToString(init_coefs):
    return "$ Initial: " + " + ".join([
        f"{init_coefs[i]:.3f}" +
        " \cdot x ^ {" + str(i) + "}"
        for i in range(len(init_coefs))]) + "$"


def func(x, coeff):
    return coeff[3] * np.cos(9*x) * x ** 3 + coeff[2] * np.sin(0.1/x ** 4) * x ** 2 + coeff[1] * x\
        + coeff[0] + coeff[4] * np.cos(x ** 2)+4*np.log(1/x**3)**2


def main():
    x = np.arange(1, 100)
    y = func(x, init_coefs)
    yn = y + NOISE * np.random.randn(len(x))

    # NonBlocking method with iterations limit
    solver = DogLeg_Met(function=func, max_iter=100000, eps=10 ** (-5), trust_reg=500)
    data = 1000000 * np.random.random(len(init_coefs))
    epoch, iters = solver.recoverCoefs(x, yn, data)
    computed = solver.get_computed_coefs()
    divergence = solver.getDivergence()

    plt.figure()
    plt.plot(x, y, label="Initial function", linewidth=2)
    plt.plot(x, yn, label="Randomized data", linewidth=2)
    plt.plot(x, computed, label="Computed: epoch " + str(epoch) + "; real_iter " + str(iters), linewidth=2)
    plt.plot(x, divergence, label="Divergence", linewidth=2)
    string_func = funcToString(init_coefs)
    plt.title(string_func)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
