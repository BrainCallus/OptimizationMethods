from methods import *
from learning_rates import *
from functions_and_gradients import *
from undone_graphics import *
from errors_functions import *
from regression_generation import *
from regularization import *

reg = Elastic()

lr = exp_learning_rate(0.05)
lr.decay = 1
gd = GD(lr=lr, regularization=reg)

lr = const_learning_rate(0.002)
momentum = Momentum(lr=lr, regularization=reg)

lr = time_learning_rate(0.001)
nag = NAG(lr=lr, regularization=reg)

lr = const_learning_rate(50)
# lr = time_learning_rate(90)
# lr = step_learning_rate(70, 20)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = exp_learning_rate(50)
rms_prop = RMSProp(lr=lr, regularization=reg)

lr = const_learning_rate(10)
adam = Adam(lr=lr, regularization=reg)

start = [10, 10, 10]
func_coefs = [100, 5, 7]
# хочется задавать вектор коэффициэнтов произвольной длины и автоматом подставлять его в лямду
xs, ys = generate_descent_polynom(10, lambda x: func_coefs[2] * x ** 2 + func_coefs[1] * x + func_coefs[0], 10, 50)
xs, ys = generate_descent_polynom(10, polynom(func_coefs), 10, 50)
xs = np.asarray(xs)
ys = np.asarray(ys)
xy = np.dstack((xs, ys))[0]

# method = momentum

error_function = BatchGD(quadratic_error_func, quadratic_error_func_grad, xy)

for method in [adam, nag, momentum, rms_prop]:
    iterations, dots = method.execute(start, error_function)

    draw_regression(method, error_function, start, xy, func_coefs)

    print(len(xs))
    print(iterations)
    print(dots[-1])
    print("---")
