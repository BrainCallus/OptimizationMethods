from OptimizationMethods.Lab2.documentation.methods import *
from OptimizationMethods.Lab2.documentation.learning_rates import *
from OptimizationMethods.Lab2.documentation.functions_and_gradients import *
from OptimizationMethods.Lab2.execute_documetation.graphics import *
from OptimizationMethods.Lab2.documentation.regularization import *

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
func = lambda x: x[0] ** 2 + 9 * x[1] ** 2
grad = lambda x: [2 * x[0], 18 * x[1]]
function = Function(func, grad, title="$x^2 + 9y^2$")

draw_levels(function, start, gd, momentum, nag, rms_prop, adam, ada_grad)
