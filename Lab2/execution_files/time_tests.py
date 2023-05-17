from OptimizationMethods.Lab2.execute_lib.excel import *
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


lr = const_learning_rate(0.1)
reg = NoRegularization()
method1 = Adam(lr=lr, regularization=reg)
reg = L1Regularization()
method2 = Adam(lr=lr, regularization=reg)
reg = L2Regularization()
method3 = Adam(lr=lr, regularization=reg)
reg = Elastic()
method4 = Adam(lr=lr, regularization=reg)


start = [-40, 45]
func = lambda x: x[0] ** 2 + 9 * x[1] ** 2 + 5
grad = lambda x: [2 * x[0], 18 * x[1]]
function = Function(func, grad, title="$x^2 + 9y^2 + 5$")

test_count = 500
# res = do_several_tests_with_consts(time_test, test_count, function, gd, momentum, nag, ada_grad, rms_prop, adam)

names = ['NoRegularization', 'L1', 'L2', 'Elastic']
res = do_several_tests_with_consts(time_test, test_count, function, names, method1, method2, method3, method4)

show_tests_graph(res, plot_type="hist",
                 title="Working time of different methods",
                 xy_names=["Method", "milliseconds"],
                 plot_comment="average of " + str(test_count) + " tests")

test_name = "time_methods"
make_excel_table(res, "./../tables/" + test_name + ".xlsx")

# OptimizationMethods/Lab2/execution_files/regularization_test.py
# OptimizationMethods/Lab2/tables