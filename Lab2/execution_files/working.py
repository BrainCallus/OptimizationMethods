from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.lib.functions_and_gradients import *
from OptimizationMethods.Lab2.lib.polynom_function import polynom
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.lib.errors_functions import *
from OptimizationMethods.Lab2.execute_lib.regression_generation import *
from OptimizationMethods.Lab2.lib.regularization import *

reg = Elastic

lr = exp_learning_rate(0.05)
lr.decay = 1
gd = GD(lr=lr, regularization=reg)

lr = exp_learning_rate(0.2)
momentum = Momentum(lr=lr, regularization=reg)

lr = time_learning_rate(0.001)
nag = NAG(lr=lr, regularization=reg)

# lr = const_learning_rate(50)
# lr = time_learning_rate(90)
lr = step_learning_rate(70, 20)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = exp_learning_rate(50)
rms_prop = RMSProp(lr=lr, regularization=reg)

lr = exp_learning_rate(95)  # 35 55
adam = Adam(lr=lr, regularization=reg)

start = [0, 0, 0, 0, 0]
func_coefs = [10, 5, -7, -3, 9]
#start = np.zeros(func_coefs.__len__())
xs, ys, y_real = generate_descent_polynom(50, polynom(func_coefs), 100)
xy = np.dstack((xs, ys))[0]
xy_real = np.dstack((xs, y_real))[0]
# method = momentum

error_function = MiniBatchGD(quadratic_error_func, quadratic_error_func_grad, xy, 20)

for method in [adam]:
    iterations, dots = method.execute(start, error_function)

    draw_regression(method, error_function, start, xy, xy_real, func_coefs)

    # print(len(xs))
    # print(iterations)
    # print(dots[-1])
    print("---")
