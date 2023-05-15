from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.lib.learning_rates import *
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.execute_lib.tests import *
from OptimizationMethods.Lab2.lib.regularization import *

reg = NoRegularization()
lr = exp_learning_rate(0.07)
method = NAG(lr=lr, regularization=reg)

n_points = 100

res = batch_size_test(method, 10, 70, n_points)
show_results(res, title="Dependence of the number of iterations on the batch size",
             xy_names=["Number of iterations", "Batch size"],
             plot_comment = method.name + " : " + str(n_points) + " points")
