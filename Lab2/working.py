from methods import *
from learning_rates import *
from functions_and_gradients import *
from undone_graphics import *
from errors_functions import *
from regression_generation import *
from regularization import *


reg = Elastic()

lr = exp_learning_rate(0.1)
gd       = GD      (lr=lr, regularization=reg)
momentum = Momentum(lr=lr, regularization=reg)
nag      = NAG     (lr=lr, regularization=reg)

lr = exp_learning_rate(50)
ada_grad = AdaGrad (lr=lr, regularization=reg)

lr = exp_learning_rate(10)
rms_prop = RMSProp (lr=lr, regularization=reg)
adam     = Adam    (lr=lr, regularization=reg)

method = gd

start = [100, 100]
func = lambda x: 3 * x[0] ** 2 + x[1] ** 2
grad = lambda x: [6 * x[0], 2 * x[1]]
function = Function(func, grad)

draw_levels(function, start, gd, momentum, nag, ada_grad)
