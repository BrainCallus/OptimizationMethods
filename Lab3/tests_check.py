import time
import os

from numpy import cos, sin, log

from Lab3.lib.BFGS import *
from Lab3.lib.batch_guys import *
from Lab3.lib.absNewton import *
from Lab3.tests import time_test
from Lab3.visual import *


# Настройки данных
def func(x, coeff):
    return np.asarray(coeff[0] * log(x) + coeff[1] * cos(x) + coeff[2] * sin(x) - coeff[3] * cos(x)**2 + coeff[4] * sin(x) ** 2)

NOISE = 10
DATA_SIZE = 100
init_coefs = [-33, 22, 100, -150, 1]
x = np.arange(1, 1 + DATA_SIZE)
y = func(x, init_coefs)
yn = y + NOISE * np.random.randn(DATA_SIZE)
initX = 10 * np.random.random(len(init_coefs))
def main():
    st = time.time_ns()

    method1 = Adam(lr=exp_learning_rate(10))
    method2 = GD(lr=const_learning_rate(0.01))
    method3 = NAG(lr=exp_learning_rate(0.001))
    method4 = Momentum(lr=exp_learning_rate(0.01))
    method5 = Golden(lr=exp_learning_rate(2))
    method6 = AdaGrad(lr=exp_learning_rate(10))
    method7 = RMSProp(lr=exp_learning_rate(10))
    mainMethod = method2  # основной метод (solver 3,4,5)

    solver1 = DogLeg_Met(function=func)
    solver2 = GN_Met(function=func)
    solver3 = Batch(function=func, method=mainMethod)
    solver4 = MiniBatch(function=func, method=mainMethod)
    solver5 = Stochastic(function=func, method=mainMethod)
    solver6 = BFGS(function=func)
    solver7 = L_BFGS(function=func)
    mainSolver = solver4  # основной солвер

    time_test(mainSolver, initX, x, yn)


if __name__ == "__main__":
    main()
    duration = 0.3  # seconds
    freq = 440 * 1.5  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
