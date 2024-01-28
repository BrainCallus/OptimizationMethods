import numpy as np

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
    mainSolver = solver7  # основной солвер

    # Массовые тесты

    # список методов, которые будут тестироваться

    solversTeam = [solver1, solver2]

    # список названий методов, которые будут в легенде

    names = ["DogLeg", "GN"]

    # Тесты по оси х

    mult_test = mult_test_noise
    # mult_test = mult_test_dimensions,
    # mult_test mult_size_data_size

    # Результаты тестов по оси у

    test = time_test
    # test = iter_test
    # test = result_norm_test
    # test = memory_test 

    # Массив данных, которые будут по оси y

    mass_param = [i + 1 for i in range(10)]

    # количество тестов (из которых будет браться среднее) 
    # на один "параметр" из прошлого массива

    test_number = 10

    # то есть, буквально, если есть 3 солвера, len(mass_param) = 2 
    # и коичество = 10, то будет проведено всего 60 тестов.

    mult_tests_visuals(
        solvers=solversTeam,
        test_function=test,
        mult_test_function=mult_test,
        names=names,
        params=mass_param,
        test_number_for_iteration=test_number,
        noise_init=10,
        #noise_real=100,
        noise=10,
        data_size=100,
        dimensions=4
    )

    # Если проходит тест по шумам, и указать параметр "noise", 
    # он ни на что не повлияет 


if __name__ == "__main__":
    main()
