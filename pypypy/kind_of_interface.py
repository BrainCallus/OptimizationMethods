from defs_grads import *
from pypypy.methods import *
from pypypy.out_functions import *
from matplotlib import pyplot as plt
from PIL import Image


def draw_function_graph(given_matrix, func, grad, lr=None, eps=None):
    np.seterr(invalid='ignore')
    np.seterr(over='ignore')
    start = [120, -120]
    if eps is None:
        eps = 0.001
    if lr is None:
        lr = 0.01

    learning_rate = gen_learning_rate(lr)
    points1 = using_grad_vector(learning_rate, start, eps, func, grad)
    xs1 = [i[0] for i in points1[1]]
    ys1 = [i[1] for i in points1[1]]
    points2 = using_grad_vector(golden, start, eps, func, grad)
    xs2 = [i[0] for i in points2[1]]
    ys2 = [i[1] for i in points2[1]]

    left = min(min(xs1), min(xs2))
    right = max(max(xs1), max(xs2))
    up = max(max(ys1), max(ys2))
    bottom = min(min(ys1), min(ys2))


    x0 = np.linspace(left - 10, right + 10, 1000)
    y0 = np.linspace(bottom - 10, up + 10, 1000)

    x, y = np.meshgrid(x0, y0)
    plt.title(function_two_out(given_matrix))
    plt.contour(x, y, func([x, y]), levels=sorted(set([func(i) for i in points1[1]])))
    plt.plot(xs1, ys1, 'o--')
    plt.plot(xs2, ys2, 'o--')

    print(points1[0])
    print(points1[1][-1], points1[2][-1])
    print(points2[0])
    print(points2[1][-1], points2[2][-1])

    plt.show()


def analyze_function(n, func, grad, vector=None):
    np.seterr(invalid='ignore')
    np.seterr(over='ignore')
    eps = 10 ** (-4)
    lr = 0.001

    print('a')

    start = [1] * n
    learning_rate = gen_learning_rate(lr)
    points1 = using_grad_vector(learning_rate, start, eps, func, grad)


    print('---')
    print("Градиентный спуск")
    print('Функ: ' + str(points1[0]))
    print('Итер: ' + str(len(points1[1])))
    print('Знач: ' + str(points1[2][-1]))

    start = vector
    points2 = using_grad_vector(golden, start, eps, func, grad)

    print('---')
    print("Золотое сечение")
    print('Функ: ' + str(points2[0]))
    print('Итер: ' + str(len(points2[1])))
    print('Знач: ' + str(points2[2][-1]))

    print('---------------')
    print('---------------')
    print('---------------')


def generate_random_function_vector(n, k):
    main_vector = generate_diagonal_1(k, n)
    func = function_generator_vector(n, main_vector)
    grad = gradient_generator_vector(n, main_vector)
    return func, grad, main_vector


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
        
        
def scaling(number, method):

    lr = gen_learning_rate(0.01)

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
            return x[0] ** 2 + x[1] ** 2 - x[0] * x[1] / math.sqrt(5)

        def grad33(x):
            return np.asarray([10 * x[0] / math.sqrt(5) - x[1], 2 * x[1] - x[0] / math.sqrt(5)])

        func3_name = '$5x^2 + y^2 - xy$'

        if method == 'golden':
            scaling_functions(func3_name, golden, f3, f33, grad3, grad33)
        else:
            scaling_functions(func3_name, lr, f3, f33, grad3, grad33)



def scaling_functions(func_name, method, func1, func2, grad1, grad2):
    np.seterr(invalid='ignore')
    np.seterr(over='ignore')
    eps = 10 ** (-4)


    start = [11, -11]
    points1 = using_grad_vector(method, start, eps, func1, grad1)
    xs1 = [i[0] for i in points1[1]]
    ys1 = [i[1] for i in points1[1]]

    start = [11, -11]
    points2 = using_grad_vector(method, start, eps, func2, grad2)
    xs2 = [i[0] for i in points2[1]]
    ys2 = [i[1] for i in points2[1]]

    left = min(min(xs1), min(xs2))
    right = max(max(xs1), max(xs2))
    up = max(max(ys1), max(ys2))
    bottom = min(min(ys1), min(ys2))

    x0 = np.linspace(left - 10, right + 10, 1000)
    y0 = np.linspace(bottom - 10, up + 10, 1000)

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


