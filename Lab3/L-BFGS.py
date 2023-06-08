import math
import numpy as np

""""
в точности функция wolfe из 1 лабы, изменено только название))
"""


def line_search(x, p, f, grad):
    nabl = grad(x)
    alf = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    new_x = x + alf * p
    new_nabl = grad(new_x)
    f_new_x = f(new_x)
    while f_new_x > fx + (c1 * alf * nabl.T @ p) \
            or math.fabs(new_nabl.T @ p) > c2 * math.fabs(nabl.T @ p) and f_new_x != fx:
        alf *= 0.5
        new_x = x + alf * p
        f_new_x = f(new_x)
        new_nabl = grad(new_x)
    return alf


""""
    тут мы, в отличие от обычного bfgs не храним матрицу, а сохраняем в очередь нужные k значений параметров
"""


def BFGS(x, eps, f, grad, queue_sz):
    steps_arg = [x]
    steps_f = [f(x)]
    sList = []  # должны выполнять роль очереди размера <= queue_sz
    yList = []
    rhoList = []
    alphaList = []
    i = 1
    nabl = grad(x)
    xPrev = x
    gradPrev = nabl - nabl
    while np.linalg.norm(nabl) > eps:
        q = grad(x)
        j = 0
        while j < rhoList.__len__():
            rho = rhoList[j]
            s = sList[j]
            y = yList[j]
            alpha = np.dot(s, q) * rho
            q = q - y * alpha
            j = j + 1
        gamma = 1.0
        if i != 1:
            gamma = np.dot(sList[0], yList[0]) / (np.dot(yList[0], yList[0]) + eps * 10 ** (-3))
        r = q * gamma
        j = rhoList.__len__() - 1
        while j >= 0:
            rho = rhoList[j]
            y = yList[j]
            s = sList[j]
            alpha = alphaList[j]
            betta = rho * np.dot(y, r)
            r = r + s * (alpha - betta)
            j = j - 1

        if rhoList.__len__() == queue_sz:
            sList.pop()
            yList.pop()
            rhoList.pop()
            alphaList.pop()
            # тут типа из очереди удаляем последний элемент для sList, yList, rhpList, alphaList
        alf = line_search(x, -r, f, nabl)
        xPrev = x
        x = x - r * alf
        nabl = grad(x)
        sList.insert(0, x - xPrev)
        yList.insert(0, nabl - gradPrev)
        rhoList.insert(0, 1.0 / (np.dot(yList[0], sList[0]) + eps * 10 ** (-3)))
        alphaList.insert(0, alf)
        gradPrev = nabl
        steps_arg.append(x)
        steps_f.append(f(x))
        i += 1
    return i, steps_arg, steps_f
