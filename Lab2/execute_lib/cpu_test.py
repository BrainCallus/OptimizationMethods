from Lab2.lib.functions_and_gradients import *
from Lab2.lib.learning_rates import *
from Lab2.lib.methods import *
from Lab2.lib.regularization import *

import sys

def execute(method_name):
    reg = NoRegularization()

    lr = exp_learning_rate(0.01)
    method = GD(lr=lr, regularization=reg)

    if method_name == 'momentum':
        lr = exp_learning_rate(0.01)
        method = Momentum(lr=lr, regularization=reg)

    if method_name == 'nag':
        lr = exp_learning_rate(0.01)
        method = NAG(lr=lr, regularization=reg)

    if method_name == 'adagrad':
        lr = step_learning_rate(10, 4)
        method = AdaGrad(lr=lr, regularization=reg)

    if method_name == 'rmsprop':
        lr = const_learning_rate(0.1)
        method = RMSProp(lr=lr, regularization=reg)

    if method_name == 'adam':
        lr = const_learning_rate(0.1)
        method = Adam(lr=lr, regularization=reg)

    start = [-40, 45]
    func = lambda x: x[0] ** 2 + 9 * x[1] ** 2 + 5
    grad = lambda x: [2 * x[0], 18 * x[1]]
    function = Function(func, grad, title="$x^2 + 9y^2 + 5$")

    method.execute(start, function)


if __name__ == "__main__":
    name = int(sys.argv[1])
    execute(name)


