import numpy as np
import math

# исследуемая функция № 1

def used_func1(x):
    return x[0] ** 2 + 13 * x[1] ** 2 + 3


def used_grad1(x):
    return np.asarray([2 * x[0], 26 * x[1]])


def used_sub_func1(x):
    return x[0] ** 2 + x[1] ** 2 + 3


def used_sub_grad1(x):
    return np.asarray([2 * x[0], 26 * x[1] / math.sqrt(13)])


used_func1_name = '$x^2 + 13y^2 + 3$'

func_set_1 = [used_func1, used_grad1, used_sub_func1, used_sub_grad1, used_func1_name]


# исследуемая функция № 2

def used_func2(x):
    return 6 * x[0] ** 2 + 7 * x[1] ** 2 - 1


def used_grad2(x):
    return np.asarray([12 * x[0], 14 * x[1]])


def used_sub_func2(x):
    return (x[0] ** 2) + (x[1] ** 2) - 1


def used_sub_grad2(x):
    return np.array([12 * x[0] / math.sqrt(6), 14 * x[1]  / math.sqrt(7)])


used_func2_name = '$6x^2 + 7y^2 - 1$'

func_set_2 = [used_func2, used_grad2, used_sub_func2, used_sub_grad2, used_func2_name]


# исследуемая функция № 3

def used_func3(x):
    return 5 * x[0] ** 2 + x[1] ** 2 - x[0] * x[1]


def used_grad3(x):
    return np.asarray([10 * x[0] - x[1], 2 * x[1] - x[0]])


def used_sub_func3(x):
    return x[0] ** 2 * 2 + x[1] ** 2 / 4.5 * 2 - x[0] * x[1] / math.sqrt(4.5 / 4)


def used_sub_grad3(x):
    return np.asarray([10 * x[0] / math.sqrt(4.5 / 2) - x[1] * math.sqrt(2),
                       2 * x[1] * math.sqrt(2) - x[0] / math.sqrt(4.5 / 2)])


used_func3_name = '$5x^2 + y^2 - xy$'

func_set_3 = [used_func3, used_grad3, used_sub_func3, used_sub_grad3, used_func3_name]


# Названия методов

method_names = ['learning_rate', 'golden', 'wolfe']
