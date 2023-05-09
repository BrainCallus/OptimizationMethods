from methods import *
from learning_rates import *
from functions_and_gradients import *
from undone_graphics import *
from errors_functions import *
from regression_generation import *
from regularization import *

reg = NoRegularization()

lr = const_learning_rate(0.1)
gd = GD(lr=lr, regularization=reg)

lr = exp_learning_rate(0.07)
momentum = Momentum(lr=lr, regularization=reg)
nag = NAG(lr=lr, regularization=reg)

# lr = exp_learning_rate(90)
# lr = time_learning_rate(90)
lr = step_learning_rate(70, 20)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = const_learning_rate(5)
rms_prop = RMSProp(lr=lr, regularization=reg)
lr = exp_learning_rate(85)
adam = Adam(lr=lr, regularization=reg)

start = [-40, 45]
func = lambda x: x[0] ** 2 + 9*x[1] ** 2
grad = lambda x: [2 * x[0], 18 * x[1]]
function = Function(func, grad)
funcToStr = "x^2 + 9y^2"

draw_levels(function, start, funcToStr, gd, momentum, nag, rms_prop, adam, ada_grad)
