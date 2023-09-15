from OptimizationMethods.Lab2.lib.methods import *
from OptimizationMethods.Lab2.execute_lib.graphics import *
from OptimizationMethods.Lab2.execute_lib.tests import *
from OptimizationMethods.Lab2.execute_lib.vis_normalization import *

lr = const_learning_rate(15)
method = Adam(lr=lr)
method.set_max_iterations(1000)

t = time.time_ns()
start = 1
finish = 1000
n_points = 1000
step = 1
tests_count = 1

res0 = do_several_tests_batch_size(tests_count, method, start, finish, step, n_points)
method.set_lr(exp_learning_rate(10))
res1 = do_several_tests_batch_size(tests_count, method, start, finish, step, n_points)
method.set_lr(time_learning_rate(10))
res2 = do_several_tests_batch_size(tests_count, method, start, finish, step, n_points)
method.set_lr(step_learning_rate(0.1, 20))
res3 = do_several_tests_batch_size(tests_count, method, start, finish, step, n_points)

print("Время исполнения:", (time.time_ns() - t) / 10 ** 9, "секунд")


show_several_graph([res0, res1, res2, res3], title="Different lr dependence of the time of work on the batch size",
                 xy_names=["Batch size", "Seconds"],
                 plot_comments = [
                     "const", "exp", "time", "step"
                 ],
                 plot_comm = "Average of " + str(tests_count) + " tests")
res = [res0, res1, res2, res3]

f = open("/home/natalia/HW/MetOpt/OptimizationMethods/Lab2/execution_files/test.txt", "w")
for j in range(len(res)):
    f.write("aaa\n")
    for i in range(len(res[j])):
        f.write(str(res[j][i][0]) + " " + str(res[j][i][1]) + "\n")
f.close()

res = [res0, res1, res2, res3]

for i in range(len(res)):
    y = np.asarray([i[1] for i in res[i]])
    x = np.asarray([i[0] for i in res[i]])
    newY = savitzky_golay_piecewise(x, y)
    res[i] = np.dstack((x, newY))[0]



show_several_graph(res, title="Different lr dependence of the time of work on the batch size",
                 xy_names=["Batch size", "Seconds"],
                 plot_comments = [
                     "const", "exp", "time", "step"
                 ],
                 plot_comm = "Average of " + str(tests_count) + " tests")

