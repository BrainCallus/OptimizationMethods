from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.lib.functions_and_gradients import *
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.lib.regularization import *

reg = NoRegularization()

lr = const_learning_rate(0.08)
gd = GD(lr=lr, regularization=reg)

lr = exp_learning_rate(0.042) # 0.094
momentum = Momentum(lr=lr, regularization=reg)
lr = exp_learning_rate(0.06)# 0.075
nag = NAG(lr=lr, regularization=reg)

# lr = exp_learning_rate(90)
# lr = time_learning_rate(90)
lr = step_learning_rate(70, 20)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = const_learning_rate(5)
rms_prop = RMSProp(lr=lr, regularization=reg)
lr = exp_learning_rate(22)
adam = Adam(lr=lr, regularization=reg)

start = [-40, 45]
func = lambda x: x[0] ** 2 + 9 * x[1] ** 2
grad = lambda x: [2 * x[0], 18 * x[1]]
function = Function(func, grad, title="$x^2 + 9y^2 + 5$")

draw_levels(function, start, gd, momentum, nag, rms_prop, adam, ada_grad)
