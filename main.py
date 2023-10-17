import numpy as np

np.set_printoptions(precision=4)
import tracemalloc
from numpy import cos, sin, log

from Lab3.lib.BFGS import *
from Lab3.lib.batch_guys import *
from Lab3.lib.absNewton import *
from Lab3.sev_tests import *
from Lab3.tests import *


def main():
    method1 = Adam(lr=exp_learning_rate(60))
    method2 = GD(lr=const_learning_rate(0.01))
    method3 = NAG(lr=exp_learning_rate(0.01))
    method4 = Momentum(lr=exp_learning_rate(0.01))
    method5 = Golden()
    method6 = AdaGrad(lr=const_learning_rate(100), max_iter=1000)
    method7 = RMSProp(lr=exp_learning_rate(10))
    mainMethod = method1  # основной метод (solver 3,4,5)

    solver1 = DogLeg_Met()
    solver2 = GN_Met()
    solver3 = Batch(method=mainMethod)
    solver4 = MiniBatch(method=mainMethod)
    solver5 = Stochastic(method=mainMethod)
    solver6 = BFGS()
    solver7 = L_BFGS()
    mainSolver = solver4  # основной солвер


    mult_tests_visuals(
    [
        solver1,
        solver2
        ], 
    result_norm_test, 
    mult_test_dimensions, 
    [i + 1 for i in range(10)],
    [
        "DogLeg",
        "GN"
        ],
    test_number_for_iteration=10
    )

if __name__ == "__main__":
    main()
