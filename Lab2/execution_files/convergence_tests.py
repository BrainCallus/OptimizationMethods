from OptimizationMethods.Lab2.execute_lib.graphics import show_tests_graph
from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.lib.functions_and_gradients import *
from OptimizationMethods.Lab2.lib.regularization import *

def convergence_test(l):
    res = []
    for i in range(len(l) - 1):
        res.append(l[i + 1] / l[i])
    return res

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
function = Function(func, grad, title="$x^2 + 9y^2$")

method = gd
cut = 100
cut2 = 100

iterations, points = method.execute(start, function)

fs = np.asarray([i[1] for i in points])

analyse1 = np.dstack((np.asarray(range(cut, iterations)), fs[cut:]))[0]

analyse2 = convergence_test(fs)
analyse2 = np.dstack((np.asarray(range(cut2, iterations - 1)), analyse2[cut2:]))[0]

show_tests_graph(analyse1, title="Analyse convergence of method",
                 xy_names=["Number of iterations", "Function value"],
                 plot_comment = method.name,
                 plot_style="-")

show_tests_graph(analyse2, title="Analyse convergence of method",
                 xy_names=["Number of iterations", "$f_i / f_{i-1}$"],
                 plot_comment = method.name + " : " + str(analyse2[-1][1]),
                 plot_style="-")

# в таком виде доказывается, что линейной сходимости НЕТ