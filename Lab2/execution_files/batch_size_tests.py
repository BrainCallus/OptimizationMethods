from OptimizationMethods.Lab2.execute_lib.excel import *
from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.execute_lib.tests import *
from OptimizationMethods.Lab2.lib.regularization import *

reg = Elastic()
lr = exp_learning_rate(0.2)
method = RMSProp(lr=lr, regularization=reg)


start = 1
finish = start + 30
n_points = 50
tests_count = 1

res = do_several_tests(batch_size_test, tests_count, method, start, finish, n_points)

print(res)

show_tests_graph(res, title="Dependence of the number of iterations on the batch size",
                 xy_names=["Number of iterations", "Batch size"],
                 plot_comment = method.name + " : " + str(n_points) + " points \n average of " + str(tests_count) + " tests",
                 plot_style="-")

test_name = "batch_iter_Nesterov_exp_lr_0-07_elastic_0_350_350_10"
make_excel_table(res, "./tables/" + test_name + ".xlsx")

