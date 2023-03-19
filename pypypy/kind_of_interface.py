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


def generate_random_function_quadratic(n, k):
    main_matrix = matrix_out_of_vector(n, generate_diagonal_1(k, n))
    func = function_generator(n, main_matrix)
    grad = gradient_generator(n, main_matrix)
    return func, grad, main_matrix

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


