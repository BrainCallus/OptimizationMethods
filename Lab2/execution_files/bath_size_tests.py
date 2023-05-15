from OptimizationMethods.Lab2.documentation.methods import *
from OptimizationMethods.Lab2.documentation.learning_rates import *
from OptimizationMethods.Lab2.execute_documetation.graphics import *
from OptimizationMethods.Lab2.execute_documetation.tests import *
from OptimizationMethods.Lab2.documentation.regularization import *

reg = Elastic()
lr = const_learning_rate(0.01)
method = NAG(lr=lr, regularization=reg)

res = batch_size_test(method, 10, 50, 50)
show_results(res, title="Зависимость количества итераций от размера батча",
             xy_names=["Размер батча", "Количество итераций"])
