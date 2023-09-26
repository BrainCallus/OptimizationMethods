import time

import numpy as np
from numpy import cos, sin, log

import matplotlib.pyplot as plt

from OptimizationMethods.Lab3.lib.BFGS import *
from OptimizationMethods.Lab3.lib.batch_guys import *
from OptimizationMethods.Lab3.lib.absNewton import *
from OptimizationMethods.Lab3.visual import funcToString


# Настройки данных
def func(x, coeff):
    return np.asarray(coeff[0] * log(x) + coeff[1] * cos(x))

NOISE = 10
DATA_SIZE = 100
init_coefs = [-33, 22]
x = np.arange(1, 1 + DATA_SIZE)
y = func(x, init_coefs)
yn = y + NOISE * np.random.randn(DATA_SIZE)
initX = 10 * np.random.random(len(init_coefs))
def main():
    st = time.time_ns()


    method1 = Adam(lr=exp_learning_rate(70))
    method2 = GD(lr=const_learning_rate(0.001))
    method3 = NAG(lr=exp_learning_rate(0.001))
    method4 = Momentum(lr=exp_learning_rate(0.01))
    method5 = Golden(lr=exp_learning_rate(1))
    method6 = AdaGrad(lr=exp_learning_rate(10))
    method7 = RMSProp(lr=exp_learning_rate(10))
    mainMethod = method7  # основной метод (solver 3,4,5)

    solver1 = DogLeg_Met(function=func)
    solver2 = GN_Met(function=func)
    solver3 = Batch(function=func, method=mainMethod)
    solver4 = MiniBatch(function=func, method=mainMethod)
    solver5 = Stochastic(function=func, method=mainMethod)
    solver6 = BFGS(function=func)
    solver7 = L_BFGS(function=func)
    mainSolver = solver4  # основной солвер

    print(initX)
    epoch, iters = mainSolver.recoverCoefs(x, yn, initX)
    computed = mainSolver.getComputedCoefficients()
    divergence = mainSolver.getDivergence()
    time1 = time.time_ns() - st

    plt.figure()
    plt.plot(x, y, label="Initial function", linewidth=2)
    plt.plot(x, yn, label="Randomized data", linewidth=2)
    plt.plot(x, computed, label="Computed: epoch " + str(epoch)
                                + "; real_iter " + str(iters), linewidth=2)
    plt.plot(x, divergence, label="Divergence", linewidth=2)
    plt.title(funcToString(init_coefs))
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
    print(mainSolver.coefficients)


if __name__ == "__main__":
    main()
