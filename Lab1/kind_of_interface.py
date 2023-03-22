import numpy as np

from defs_grads import *
from pypypy.methods import *
from pypypy.out_functions import *
from analysed_funcs import *
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

def analyze_function(n, func, grad, given_vector=None, vector_using=None):
    if vector_using is None:
        vector_using = [True, True, True]
    np.seterr(invalid='ignore')
    np.seterr(over='ignore')
    np.seterr(divide='ignore')
    alpha = 5
    start = [alpha] * n
    eps = 10 ** (-4)
    lr = 0.001

    if vector_using[0]:
        learning_rate = gen_learning_rate(lr)
        points1 = using_grad_vector(learning_rate, start, eps, func, grad)
    else:
        points1 = None

    if vector_using[2]:
        points3 = wolfe(start, 10**-3 * 2, func, grad)
    else:
        points3 = None

    if vector_using[1]:
        if given_vector is not None:
            start = given_vector
        points2 = using_grad_vector(golden, start, eps, func, grad)
    else:
        points2 = None

    return points1, points2, points3


def single_analyse_result(n, func, grad, given_vector=None):
    points1, points2, points3 = analyze_function(n, func, grad, given_vector)

    print('---')
    print("Градиентный спуск")
    print('Функ: ' + str(points1[0]))
    print('Итер: ' + str(len(points1[1])))
    print('Знач: ' + str(points1[2][-1]))

    print('---')
    print("Золотое сечение")
    print('Функ: ' + str(points2[0]))
    print('Итер: ' + str(len(points2[1])))
    print('Знач: ' + str(points2[2][-1]))

    print('---')
    print("Вульфе")
    print('Функ: ' + str(points3[0]))
    print('Итер: ' + str(len(points3[1])))
    print('Знач: ' + str(points3[2][-1]))

    print('---------------')


def draw_function_graph(func, grad, given_matrix=None, given_vector=None):
    n = 2
    points1, points2, points3 = analyze_function(n, func, grad)

    xs1 = [i[0] for i in points1[1]]
    ys1 = [i[1] for i in points1[1]]

    xs2 = [i[0] for i in points2[1]]
    ys2 = [i[1] for i in points2[1]]

    xs3 = [i[0] for i in points3[1]]
    ys3 = [i[1] for i in points3[1]]

    left = min(min(xs1), min(xs2), min(xs3))
    right = max(max(xs1), max(xs2), max(xs3))
    up = max(max(ys1), max(ys2), max(ys3))
    bottom = min(min(ys1), min(ys2), min(ys3))

    x0 = np.linspace(left - 1, right + 1, 100)
    y0 = np.linspace(bottom - 1, up + 1, 100)

    x, y = np.meshgrid(x0, y0)
    if given_matrix is not None:
        plt.title(function_two_out(given_matrix))
    if given_vector is not None:
        plt.title(function_two_out(matrix_out_of_vector(2, given_vector)))

    ax = plt.subplot()
    ax.contour(x, y, func([x, y]), levels=sorted(set([func(i) for i in points1[1]])))
    ax.plot(xs1, ys1, 'o-', label = 'learning_rate: ' + str(len(points1[1])))
    ax.plot(xs2, ys2, 'o-', label = 'golden       : ' + str(len(points2[1])))
    ax.plot(xs3, ys3, 'o-', label = 'wolfe        : ' + str(len(points3[1])))
    ax.legend(prop='monospace')

    plt.show()


def multiple_tests_to_excel():
    file = './tables/multiple_results.xlsx'
    sheet = 'Испытания'

    ii = [1, 2, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450,
          500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    jj = [2, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450,
         500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    # jj = [2, 5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    frame = 2
    pause = 2
    methods = 1
    times = 25

    rows = frame + len(ii) + 100
    cols = frame * methods + (methods-1) * pause + methods * len(jj) + 100

    table = [['' for _ in range(rows)] for _ in range(cols)]

    table[0][0] = 'learning_rate'
    table[0][1] = 'n'
    table[1][0] = 'k'

    # table[frame + pause + len(jj)][0] = 'golden'
    # table[frame + pause + len(jj)][1] = 'n'
    # table[frame + pause + len(jj) + 1][0] = 'k'
    #
    # table[2 * frame + 2 * pause + 2 * len(jj)][0] = 'wolfe'
    # table[2 * frame + 2 * pause + 2 * len(jj)][1] = 'n'
    # table[2 * frame + 2 * pause + 2 * len(jj) + 1][0] = 'k'

    it = -1
    for j in jj:
        it += 1
        for k in range(methods):
            table[it + frame * (k + 1) + pause * k + len(jj) * k][1] = j

    it = -1
    for i in ii:
        it += 1
        table[1][it + frame] = i


    it = 0
    iter1 = -1
    for i in ii:
        iter1 += 1
        iter2 = -1
        for j in jj:
            dimensions = i
            self_number = j
            iter2 += 1
            average = np.zeros(methods)
            for k in range(times):
                a, func, grad = generate_random_function_vector(dimensions, self_number)
                points1, points2, points3 = \
                    analyze_function(dimensions, func, grad, vector_using=[True, False, False])
                average[0] += len(points1[1])
                # average[1] += len(points2[1])
                # average[2] += len(points3[1])
            average[0] = average[0] / times
            # average[1] = average[1] / times
            # average[2] = average[2] / times
            for k in range(methods):
                table[iter2 + frame * (k + 1) + pause * k + len(jj) * k][iter1 + frame] = average[k]
            it += 1
            print(it)
            print(average[0])

    df = pd.DataFrame(table)
    df.to_excel(file, sheet_name=sheet)


def functions_compare(func1, grad1, func2, grad2, method=1, func_name=None):

    vector_using = [False, False, False]
    vector_using[method - 1] = True
    method_name = method_names[method - 1]

    p1 = analyze_function(2, func1, grad1, vector_using=vector_using)
    p2 = analyze_function(2, func2, grad2, vector_using=vector_using)
    points1 = p1[method - 1]
    points2 = p2[method - 1]


    np.seterr(invalid='ignore')
    np.seterr(over='ignore')

    xs1 = [i[0] for i in points1[1]]
    ys1 = [i[1] for i in points1[1]]

    xs2 = [i[0] for i in points2[1]]
    ys2 = [i[1] for i in points2[1]]

    left = min(min(xs1), min(xs2))
    right = max(max(xs1), max(xs2))
    up = max(max(ys1), max(ys2))
    bottom = min(min(ys1), min(ys2))

    x0 = np.linspace(left - 1, right + 1, 1000)
    y0 = np.linspace(bottom - 1, up + 1, 1000)

    x, y = np.meshgrid(x0, y0)
    plt.title(func_name)
    ax = plt.subplot()
    ax.contour(x, y, func1([x, y]), levels=sorted(set([func1(i) for i in points1[1]])))
    ax.contour(x, y, func2([x, y]), levels=sorted(set([func2(i) for i in points2[1]])))
    ax.plot(xs1, ys1, 'o-', label = method_name + ': ' + str(len(points1[1])))
    ax.plot(xs2, ys2, 'o-', label = method_name + ': ' + str(len(points2[1])))
    ax.legend(prop='monospace')

    plt.show()


def scale_function(n, func, scaling_vector):
    def scaled_func(vector):
        result = [0 for _ in range(n)]
        for i in range(n):
            result[i] = vector[i] * scaling_vector[i]
        return func(result)
    return scaled_func


def generate_random_function_vector(n, k):
    main_vector = generate_diagonal_1(n, k)
    func = function_generator_vector(n, main_vector)
    grad = gradient_generator_vector(n, main_vector)
    return main_vector, func, grad


def draw_random_function(k):
    a, func, grad = generate_random_function_vector(2, k)
    draw_function_graph(func, grad, given_vector=a)


def analyze_random_function(n, k):
    a, func, grad = generate_random_function_vector(n, k)
    single_analyse_result(n, func, grad, a)


def analyse_eps_parameter(func, grad):
    start = [-15, 15]
    lr = gen_learning_rate(0.06)

    func_calls, p1, p2 = method_mas(lr, start, 9, func, grad)

    print('---------------')
    print('---------------')
    print('---------------')
    print('learning_rate')
    for i in range(len(p1)):
        # print('---')
        # print('eps=1e-0' + str(i + 1))
        # print('Функ: ' + str(func_calls[i]))
        # print('Итер: ' + str(len(p1[i])))
        # print('Рез : ' + str(p2[i][-1]))
        print(str(func_calls[i]))

    func_calls, p1, p2 = method_mas(golden, start, 9, func, grad)

    print('---------------')
    print('---------------')
    print('---------------')
    print('golden')
    for i in range(len(p1)):
        # print('---')
        # print('eps=1e-0' + str(i + 1))
        # print('Функ: ' + str(func_calls[i]))
        # print('Итер: ' + str(len(p1[i])))
        # print('Рез : ' + str(p2[i][-1]))
        print(str(func_calls[i]))


def graph_show_or_save(func, file_name=None, dir_name='graphs', save=True, show=False):
    if file_name is not None:
        file_name = './' + dir_name + '/' + file_name + '.jpg'

    x = np.linspace(-20, 20, 1000)
    x, y = np.meshgrid(x, x)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(x, y, func([x, y]))

    if save:
        plt.savefig(file_name)
        Image.open(file_name).save(file_name, 'JPEG')
    if show:
        plt.show()


