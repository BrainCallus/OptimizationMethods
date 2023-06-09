import time
import matplotlib.pyplot as plt

from OptimizationMethods.Lab3.regression.batch_guys import *

NOISE = 100
init_coefs = [-33, 14, 0.55]


def funcToString(init_coefs):
    return "$ Initial: " + " + ".join([
        f"{init_coefs[i]:.3f}" +
        " \cdot x ^ {" + str(i) + "}"
        for i in range(len(init_coefs))]) + "$"


def func(x, coeff):
    return coeff[2] * x ** 2 + coeff[1] * x\
        + coeff[0]


def main():
    r = 99
    x = np.arange(1, 1 + r)
    y = func(x, init_coefs)
    yn = y + NOISE * np.random.randn(r)
    data = 10 * np.random.random(len(init_coefs))

    st = time.time_ns()
    solver1 = MiniBatch(function=func, max_iter=10000, eps=10 ** (-5))
    epoch1, iters1 = solver1.recoverCoefs(x, yn, data)
    computed1 = solver1.get_computed_coefs()
    divergence1 = solver1.getDivergence()
    time1 = time.time_ns() - st

    string_func = funcToString(init_coefs)

    plt.figure()
    plt.plot(x, y, label="Initial function", linewidth=2)
    plt.plot(x, yn, label="Randomized data", linewidth=2)
    plt.plot(x, computed1, label="Computed: epoch " + str(epoch1) + "; real_iter " + str(iters1), linewidth=2)
    plt.plot(x, divergence1, label="Divergence", linewidth=2)
    plt.title(string_func)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()

    print(init_coefs)
    print()
    print(time1)
    print(solver1.coefficients)
    print(solver1.coefficients / init_coefs)

if __name__ == "__main__":
    main()
