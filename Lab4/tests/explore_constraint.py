import numpy as np
import scipy
from functools import partial

from Lab4.util.run_functions import run_and_return_result


# хуй знает, думаю, достаточно показать, что работает

def constraint_test(start, func):
    # y >= 0.5x (перекрываем долину в функции)
    c1 = scipy.optimize.LinearConstraint(np.array([0.5, -1]), -np.inf, 0)
    # y <= 2x (подталкиваем к точке минимума)
    c2 = scipy.optimize.LinearConstraint(np.array([2, -1]), 0, np.inf)
    # y <= x (проходим по точке минимума)
    c3 = scipy.optimize.LinearConstraint(np.array([1, -1]), 0, np.inf)
    # y >= x (проходим по точке минимума)
    c4 = scipy.optimize.LinearConstraint(np.array([1, -1]), -np.inf, 0)
    # y = x (проходим по точке минимума)
    c5 = scipy.optimize.LinearConstraint(np.array([1, -1]), 0, 0)

    # круг с радиусом 1 от (1, 1)
    nlc1 = scipy.optimize.NonlinearConstraint(lambda x: (x[0] - 1) ** 2 + (x[1] - 1) ** 2, 0, 1)

    # круг с радиусом 1 от (0, 1)
    nlc2 = scipy.optimize.NonlinearConstraint(lambda x: (x[0]) ** 2 + (x[1] - 1) ** 2, 0, 1)

    # круг с радиусом 1 от (0, 0)
    nlc3 = scipy.optimize.NonlinearConstraint(lambda x: (x[0]) ** 2 + (x[1]) ** 2, 0, 1)

    # вне круга с радиусом 1 от (1, 1)
    nlc4 = scipy.optimize.NonlinearConstraint(lambda x: (x[0] - 1) ** 2 + (x[1] - 1) ** 2, 1, np.inf)

    # вне круга с радиусом 1 от (0, 0)
    nlc5 = scipy.optimize.NonlinearConstraint(lambda x: (x[0]) ** 2 + (x[1]) ** 2, 1, np.inf)

    all_regrs = [
        (partial(scipy.optimize.minimize), "No constraints"),
        (partial(scipy.optimize.minimize, constraints=c1), "y >= 0.5x //перекрываем долину в функции"),
        (partial(scipy.optimize.minimize, constraints=c2), "y <= 2x //подталкиваем к точке минимума"),
        (partial(scipy.optimize.minimize, constraints=c3), "y <= x //проходим по точке минимума"),
        (partial(scipy.optimize.minimize, constraints=c4), "y >= x //проходим по точке минимума"),
        (partial(scipy.optimize.minimize, constraints=c5), "y = x //проходим по точке минимума"),
        (partial(scipy.optimize.minimize, constraints=nlc1), "B((1, 1), 1) //круг с радиусом 1 от (1, 1)"),
        (partial(scipy.optimize.minimize, constraints=nlc2), "B((0, 1), 1) //круг с радиусом 1 от (0, 1)"),
        (partial(scipy.optimize.minimize, constraints=nlc3), "B((0, 0), 1) //круг с радиусом 1 от (0, 0)"),
        (partial(scipy.optimize.minimize, constraints=nlc4), "not B((1, 1), 1) //вне круга с радиусом 1 от (1, 1)"),
        (partial(scipy.optimize.minimize, constraints=nlc5), "not B((0, 0), 1) //вне круга с радиусом 1 от (0, 0)"),
    ]

    for regr, name in all_regrs:
        result = run_and_return_result(lambda: regr(func, start))
        print(
            f"{name}:\n\tx = {result.result.x}\n\titers = {result.result.nit}\n\t"
            + f"time = {result.time_usage}\n\tmemory = {result.memory_usage}")


def main():
    start = [-30, 50]

    def rosenbrock(x):  # Rosenbrock function
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    print("Rosenbrock function: 100(y-x^2)^2+(1-x^2)")
    constraint_test(start, rosenbrock)
    print('=====================================\n')
    print('(x-1)^2-7xy+(y-1)^4')
    constraint_test(start, lambda x: (x[0] - 1) ** 2 - 7 * x[0] * x[1] + (x[1] - 1) ** 4)


if __name__ == "__main__":
    main()
