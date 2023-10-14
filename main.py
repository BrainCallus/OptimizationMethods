import numpy as np
np.set_printoptions(precision=4)
from numpy import cos, sin, log

from Lab3.lib.BFGS import *
from Lab3.lib.batch_guys import *
from Lab3.lib.absNewton import *
from Lab3.sev_tests import *
from Lab3.tests import *
from Lab3.visual import *


# Настройки данных
def func(x, coeff):
    return np.asarray(coeff[0] * log(x) + coeff[1] * cos(x) + coeff[2] * sin(x) - coeff[3] * cos(x)**2)

def main():
    method1 = Adam(lr=exp_learning_rate(60))
    method2 = GD(lr=const_learning_rate(0.01))
    method3 = NAG(lr=exp_learning_rate(0.01))
    method4 = Momentum(lr=exp_learning_rate(0.01))
    method5 = Golden()
    method6 = AdaGrad(lr=const_learning_rate(100), max_iter=1000)
    method7 = RMSProp(lr=exp_learning_rate(10))
    mainMethod = method7  # основной метод (solver 3,4,5)

    solver1 = DogLeg_Met(function=func)
    solver2 = GN_Met(function=func)
    solver3 = Batch(function=func, method=mainMethod)
    solver4 = MiniBatch(function=func, method=mainMethod)
    solver5 = Stochastic(function=func, method=mainMethod)
    solver6 = BFGS(function=func)
    solver7 = L_BFGS(function=func)
    mainSolver = solver5 # основной солвер

    solversTeam = [solver1, solver2]

    mult_tests_visuals(
        solversTeam, 
        result_norm_test, 
        mult_test_noise, 
        [i + 1 for i in range(100)],
        ["DogLeg", "GN"],
        test_number_for_iteration=10
        )

    mult_tests_visuals(
        solversTeam, 
        result_norm_test, 
        mult_test_data_size, 
        [(i + 1) * 500 for i in range(10)],
        ["DogLeg", "GN"],
        test_number_for_iteration=100
        )


if __name__ == "__main__":
    main()
