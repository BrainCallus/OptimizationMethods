from OptimizationMethods.Lab2.execute_lib.excel import *
from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.execute_lib.tests import *
from OptimizationMethods.Lab2.lib.regularization import *

reg = Elastic()
lr = exp_learning_rate(5)
method = RMSProp(lr=lr, regularization=reg)


start = 10
finish = start + 20
n_points = 50
tests_count = 50

res = do_several_tests_batch_size(tests_count, method, start, finish, n_points)

show_tests_graph(res, title="Dependence of the number of iterations on the batch size",
                 xy_names=["Batch size", "Number of iterations"],
                 plot_comment = method.name + " : " + str(n_points) + " points \n average of " + str(tests_count) + " tests",
                 plot_style="-")

# test_name = "batch_iter"
# make_excel_table(res, "./../tables/" + test_name + ".xlsx")


# 0
# 14431.2412109375
# 1
# 49781.282470703125
# 2
# 13683.301025390625
# 3
# 39658.4423828125
# 4
