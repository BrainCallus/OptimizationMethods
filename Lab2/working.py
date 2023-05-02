from methods import *
from learning_rates import *
from functions_and_gradients import *
from undone_graphics import *
from errors_functions import *
from regression_generation import *
from regularization import *



# Можно потыкать разные learning rate:
# const : initial_rate
# time  : initial_rate / (decay * (iter + 1))
# exp   : initial_rate * exp(- decay * iter)

# step  : initial_rate * decay ^ { (1 + iter) // epoch }
# нужен второй параметр

# Также доступны все виды регуляризации:
# NoRegularization
# L1Regularization (alpha=0.025)
# L2Regularization (beta=0.08)
# Elastic          (alpha=0.025, beta=0.08)

reg = Elastic()

lr = exp_learning_rate(0.01)
gd       = GD      (lr=lr, regularization=reg)
momentum = Momentum(lr=lr, regularization=reg)
nag      = NAG     (lr=lr, regularization=reg)

lr = exp_learning_rate(100)
ada_grad = AdaGrad (lr=lr, regularization=reg)

lr = exp_learning_rate(10)
rms_prop = RMSProp (lr=lr, regularization=reg)
adam     = Adam    (lr=lr, regularization=reg)

# другие методы объявлены, можно использовать и их


# Блок 1. Стохастический-ебанистический

# пока только линейная: функцию ошибки не переписывала

xs, ys = generate_descent_polynom(100, lambda a: 5 * a + 10, 5)
xs = np.asarray(xs)
ys = np.asarray(ys)
xy = np.dstack((xs, ys))[0]

# StochasticGD (func, grad, data)
# BatchGD      (func, grad, data)
# MiniBatchGD  (func, grad, data, batch_size)

method = momentum

error_function = BatchGD(error_func, error_func_grad, xy)
iterations, dots = method.execute([100, 100], error_function)
draw_regression(xy, dots[-1][0])

print(len(xs))
print(iterations)
print(dots[-1][0])

method = ada_grad

error_function = BatchGD(error_func, error_func_grad, xy)
iterations, dots = method.execute([100, 100], error_function)
draw_regression(xy, dots[-1][0])

print(iterations)
print(dots[-1][0])


# # Блок 2. Депрессивно-строительный
# # грустные примитивные графики с 2 переменными
#
# func = lambda x: 5 * x[0] ** 2 + 6 * x[1] ** 2 + x[0]
# grad = lambda x: [10 * x[0] + 1, 13 * x[1]]
# function = Function(func, grad)
# output_LRate(method, function)
#
#
# # Блок 3. Моно-депрессивный
# # грустные примитивные графики с 1 переменной
#
# func = lambda x: 5 * x[0] ** 2 + x[0]
# grad = lambda x: [10 * x[0] + 1]
# function = Function(func, grad)
# drawGraph(method, function)
