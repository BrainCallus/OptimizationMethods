from defs_grads import *
from pypypy.methods import *
from pypypy.out_functions import *
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

    df = pd.DataFrame(table)
    df.to_excel(file, sheet_name=sheet)


def scaling(number, method):

    lr = gen_learning_rate(0.02)

    if number == 1:
        def f1(x):
            return x[0] ** 2 + 13 * x[1] ** 2 + 3

        def f11(x):
            return x[0] ** 2 + x[1] ** 2 + 3

        def grad1(x):
          return np.asarray([2 * x[0], 26 * x[1]])

        def grad11(x):
          return np.asarray([2 * x[0], 26 * x[1] / math.sqrt(13)])


        func1_name = '$x^2 + 13y^2 + 3$'

        if method == 'golden':
            scaling_functions(func1_name, golden, f1, f11, grad1, grad11)
        else:
            scaling_functions(func1_name, lr, f1, f11, grad1, grad11)


    if number == 2:
        def f2(x):
            return 6 * x[0] ** 2 + 7 * x[1] ** 2 - 1

        def grad2(x):
          return np.asarray([12 * x[0], 14 * x[1]])

        def f22(x):
            return (x[0] ** 2) / 7 + (x[1] ** 2) / 6 - 1

        def grad22(x):
          return np.array([12 * x[0], 14 * x[1]]) / math.sqrt(7 * 6)

        func2_name = '$6x^2 + 7y^2 - 1$'

        if method == 'golden':
            scaling_functions(func2_name, golden, f2, f22, grad2, grad22)

        else:
            scaling_functions(func2_name, lr, f2, f22, grad2, grad22)

    if number == 3:
        def f3(x):
            return 5 * x[0] ** 2 + x[1] ** 2 - x[0] * x[1]

        def grad3(x):
            return np.asarray([10 * x[0] - x[1], 2 * x[1] - x[0]])

        def f33(x):
            return x[0] ** 2 * 2 + x[1] ** 2 / 4.5 * 2 - x[0] * x[1] / math.sqrt(4.5 / 4)

        def grad33(x):
            return np.asarray([10 * x[0] / math.sqrt(4.5 / 2) - x[1] * math.sqrt(2),
                               2 * x[1]* math.sqrt(2) - x[0] / math.sqrt(4.5 / 2)])

        func3_name = '$5x^2 + y^2 - xy$'

        if method == 'golden':
            scaling_functions(func3_name, golden, f3, f33, grad3, grad33)
        else:
            scaling_functions(func3_name, lr, f3, f33, grad3, grad33)


def scaling_functions(func_name, method, func1, func2, grad1, grad2):
    np.seterr(invalid='ignore')
    np.seterr(over='ignore')
    eps = 10 ** (-3)

    alpha = 23
    start = [alpha, -alpha]

    points1 = using_grad_vector(method, start, eps, func1, grad1)
    xs1 = [i[0] for i in points1[1]]
    ys1 = [i[1] for i in points1[1]]

    points2 = using_grad_vector(method, start, eps, func2, grad2)
    xs2 = [i[0] for i in points2[1]]
    ys2 = [i[1] for i in points2[1]]

    left = min(min(xs1), min(xs2))
    right = max(max(xs1), max(xs2))
    up = max(max(ys1), max(ys2))
    bottom = min(min(ys1), min(ys2))

    x0 = np.linspace(left - alpha / 10, right + alpha / 10, 1000)
    y0 = np.linspace(bottom - alpha / 10, up + alpha / 10, 1000)

    x, y = np.meshgrid(x0, y0)
    plt.title(func_name)
    plt.contour(x, y, func1([x, y]), levels=sorted(set([func1(i) for i in points1[1]])))
    plt.contour(x, y, func2([x, y]), levels=sorted(set([func2(i) for i in points2[1]])))
    plt.plot(xs1, ys1, 'o-')
    plt.plot(xs2, ys2, 'o-')

    print(points1[0])
    print(points1[1][-1], points1[2][-1])
    print(points2[0])
    print(points2[1][-1], points2[2][-1])

    plt.show()


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
