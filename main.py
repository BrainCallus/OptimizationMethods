import numpy as np
np.set_printoptions(precision=4)
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
    mainMethod = method7  # основной метод (solver 3,4,5)

    solver1 = DogLeg_Met()
    solver2 = GN_Met()
    solver3 = Batch(method=mainMethod)
    solver4 = MiniBatch(method=mainMethod)
    solver5 = Stochastic(method=mainMethod)
    solver6 = BFGS()
    solver7 = L_BFGS()
    mainSolver = solver1 # основной солвер

    solversTeam = [
        solver2,
        Stochastic(method=method1),
        Stochastic(method=method2),
        Stochastic(method=method3),
        Stochastic(method=method4),
        Stochastic(method=method5),
        Stochastic(method=method6),
        Stochastic(method=method7)
        ]

    names = [
        "Newton-Gauss",
        "Adam",
        "Gradient descent",
        "Nesterov",
        "Momentum",
        "Golden",
        "Adagrad",
        "RMSProp"
    ]

    mult_tests_visuals(
    solversTeam, 
    result_norm_test, 
    mult_test_noise, 
    [10*(i + 1) for i in range(5)],
    names,
    test_number_for_iteration=100,
    data_size=10
    )

if __name__ == "__main__":
    main()
