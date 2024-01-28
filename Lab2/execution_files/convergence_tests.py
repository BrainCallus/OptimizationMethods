from Lab2.lib.methods import *
from Lab2.lib.learning_rates import *
from Lab2.lib.functions_and_gradients import *
from Lab2.lib.regularization import *


def convergence_test(l):
    res = []
    for i in range(len(l) - 1):
        res.append(l[i + 1] / l[i])
    return res


def mini_convergence_test(l):
    return l[-1] / l[-2]


reg = NoRegularization()

lr = exp_learning_rate(0.1)
gd = GD(lr=lr, regularization=reg)

lr = exp_learning_rate(0.1)
momentum = Momentum(lr=lr, regularization=reg)

lr = exp_learning_rate(0.1)
nag = NAG(lr=lr, regularization=reg)

lr = step_learning_rate(70, 20)
ada_grad = AdaGrad(lr=lr, regularization=reg)

lr = const_learning_rate(0.1)
rms_prop = RMSProp(lr=lr, regularization=reg)

lr = const_learning_rate(0.1)
adam = Adam(lr=lr, regularization=reg)

func = lambda x: x[0] ** 2 + 9 * x[1] ** 2 + 5
grad = lambda x: [2 * x[0], 18 * x[1]]
function = Function(func, grad, title="$x^2 + 9y^2$")

method = adam

res = []

# for i in range(1, 201):
#     start = [i, i]
#     iterations, points = method.execute(start, function)
#     fs = np.asarray([k[1] for k in points])
#     res.append([i, mini_convergence_test(fs)])
#
# show_tests_graph(res, title="Dependency of linear convergence parameter q on distance from \n"
#                             "starting point to minimum point",
#                  xy_names=["$r / \sqrt{2}$", "q"],
#                  plot_comment = method.name + " : " + str(res[-1][1]),
#                  plot_style="-")
#
# test_name = "convergence_tests"
# make_excel_table(res, "./../tables/" + test_name + ".xlsx")

# -----

# for i in range(1, 51):
#     res.append([])
#     for j in range(0, 51):
#         start = [i, j]
#         iterations, points = method.execute(start, function)
#         fs = np.asarray([k[1] for k in points])
#         res[i - 1].append(mini_convergence_test(fs))
#
# show_3d_plot(res, title="Dependency of linear convergence parameter q on points",
#                  plot_comment = method.name)

# ---

# start = [35, 35]
# iterations, points = method.execute(start, function)
# fs = np.asarray([i[1] for i in points])
#
# cut = 0
# cut2 = 0
#
# print(points)
#
# analyse1 = np.dstack((np.asarray(range(cut, iterations)), fs[cut:]))[0]
# analyse2 = convergence_test(fs)
# analyse2 = np.dstack((np.asarray(range(cut2, iterations - 1)), analyse2[cut2:]))[0]
#
# show_tests_graph(analyse1, title="Analyse convergence of method",
#                  xy_names=["Number of iterations", "Function value"],
#                  plot_comment = method.name,
#                  plot_style="-")
#
# show_tests_graph(analyse2, title="Analyse convergence of method",
#                  xy_names=["Number of iterations", "$f_i / f_{i-1}$"],
#                  plot_comment = method.name + " : " + str(analyse2[-1][1]),
#                  plot_style="-")
