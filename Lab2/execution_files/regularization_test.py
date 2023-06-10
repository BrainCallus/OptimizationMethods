from Lab2.execute_lib.tests import *
from Lab2.execute_lib.graphics import *
from Lab2.execute_lib.excel import *
from Lab2.lib.functions_and_gradients import *
from Lab2.lib.learning_rates import *
from Lab2.lib.methods import *
from Lab2.lib.regularization import *


reg = NoRegularization()

lr = exp_learning_rate(0.01)
gd = GD(lr=lr, regularization=reg)

lr = exp_learning_rate(0.01)
momentum = Momentum(lr=lr, regularization=reg)

lr = exp_learning_rate(0.01)
nag = NAG(lr=lr, regularization=reg)

lr = step_learning_rate(70, 20)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = const_learning_rate(0.1)
rms_prop = RMSProp(lr=lr, regularization=reg)

lr = exp_learning_rate(95)
adam = Adam(lr=lr, regularization=reg)


start = [-5, 2]
func = lambda x: x[0] ** 2 + 9 * x[1] ** 2 + 5
grad = lambda x: [2 * x[0], 18 * x[1]]
function = Function(func, grad, title="$x^2 + 9y^2 + 5$")

method = adam

number_iter = 1

res = do_several_tests_with_consts(regularization_test, number_iter, function, method, 5)

# вот формула
# res = (res / mm - 1) * 10 ** 6

show_tests_graph(res, plot_type="hist",
                 title="Different regularization return",
                 xy_names=["Regularization type", "Result"],
                 plot_style='hist')

test_name = "regularization_tests"
make_excel_table(res, "./../tables/" + test_name + ".xlsx")
