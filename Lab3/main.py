import time
from numpy import cos, sin, log2

import matplotlib.pyplot as plt

from OptimizationMethods.Lab3.BFGS import *
from OptimizationMethods.Lab3.lib.batch_guys import *
from OptimizationMethods.Lab3.lib.absNewton import *

NOISE = 1000
DATA_SIZE = 100000
init_coefs = [-33, 22, 0.1]

def func(x, coeff):
    return coeff[0] * cos(x) + coeff[1] * sin(x**2) + coeff[2] * x

def funcToString(init_coefs):
    return "$ Initial: " + " + ".join([
        f"{init_coefs[i]:.3f}" +
        " \cdot x ^{" + str(i) + "}"
        for i in range(len(init_coefs))]) + "$"

def main():
    x = np.arange(1, 1 + DATA_SIZE)
    y = func(x, init_coefs)
    yn = y + NOISE * np.random.randn(DATA_SIZE)
    data = 10 * np.random.random(len(init_coefs))

    st = time.time_ns()
    method1 = Adam()
    method2 = GD()
    method3 = NAG()
    method4 = Momentum()
    method5 = Golden()
    mainMethod = method4 # основной метод

    solver1 = DogLeg_Met(function=func)
    solver2 = GN_Met(function=func)
    solver3 = Batch(function=func, method=mainMethod)
    solver4 = MiniBatch(function=func, method=mainMethod)
    solver5 = Stochastic(function=func, method=mainMethod)
    solver6 = BFGS(function=func)
    solver7 = L_BFGS(function=func)
    mainSolver = solver7 # основной солвер

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

    print("Время выполнения")
    print(time1 / 1000000000)
    print("Инит")
    print(init_coefs)
    print("Полученные коэффициенты")
    print(solver1.coefficients)

if __name__ == "__main__":
    main()
