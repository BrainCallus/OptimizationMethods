import numpy as np

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

#lr = const_learning_rate(50)
# lr = time_learning_rate(90)
lr = step_learning_rate(70, 20)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = exp_learning_rate(50)
rms_prop = RMSProp(lr=lr, regularization=reg)

lr = exp_learning_rate(85)
adam = Adam(lr=lr, regularization=reg)

start = [0, 0, 0,0,0]
func_coefs = [10, 5, 7,3,9]
xs, ys, y_real = generate_descent_polynom(10, polynom(func_coefs), 10, 50)
xs = np.asarray(xs)
ys = np.asarray(ys)
y_real = np.asarray(y_real)
xy = np.dstack((xs, ys))[0]
xy_real  = np.dstack((xs,y_real))[0]
# method = momentum

error_function = BatchGD(quadratic_error_func, quadratic_error_func_grad, xy)

for method in [adam,momentum,ada_grad, rms_prop,nag]:
    iterations, dots = method.execute(start, error_function)

    draw_regression(method, error_function, start, xy, xy_real, func_coefs)

    print(len(xs))
    print(iterations)
    print(dots[-1])
    print("---")
