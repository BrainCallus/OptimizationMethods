from OptimizationMethods.Lab2.execute_lib.excel import *
from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.execute_lib.tests import *
from OptimizationMethods.Lab2.lib.regularization import *

reg = Elastic()
lr = exp_learning_rate(0.07)
method = NAG(lr=lr, regularization=reg)


# старт и финиш считаются с 0 до n включительно

start = 0
finish = 30
n_points = 30
tests_count = 5

res = do_several_tests(batch_size_test, tests_count, method, start, finish, n_points)

show_tests_graph(res, title="Dependence of the number of iterations on the batch size",
                 xy_names=["Number of iterations", "Batch size"],
                 plot_comment = method.name + " : " + str(n_points) + " points \n average of " + str(tests_count) + " tests")

test_name = "batch_iter"
make_excel_table(res, "./tables/" + test_name + ".xlsx")

