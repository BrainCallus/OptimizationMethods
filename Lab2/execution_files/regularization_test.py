from OptimizationMethods.Lab2.execute_lib.tests import *
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.lib.functions_and_gradients import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.regularization import *


reg = NoRegularization()

lr = exp_learning_rate(0.01)
gd = GD(lr=lr, regularization=reg)

lr = exp_learning_rate(0.01)
momentum = Momentum(lr=lr, regularization=reg)

lr = exp_learning_rate(0.01)
nag = NAG(lr=lr, regularization=reg)

lr = step_learning_rate(10, 4)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = const_learning_rate(0.1)
rms_prop = RMSProp(lr=lr, regularization=reg)

lr = const_learning_rate(0.1)
adam = Adam(lr=lr, regularization=reg)


start = [-40, 45]
func = lambda x: x[0] ** 2 + 9 * x[1] ** 2 + 5
grad = lambda x: [2 * x[0], 18 * x[1]]
function = Function(func, grad, title="$x^2 + 9y^2 + 5$")

method = gd

res = regularization_test(function, start, method)


show_tests_graph(res, plot_type="hist",
                 title="Different regularization return",
                 xy_names=["Regularization type", "Result"],
                 plot_comment=method.name + " : " + function.get_title(),
                 plot_style='hist')
