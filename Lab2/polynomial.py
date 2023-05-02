# это более-менее рабочая версия
import numpy as np
import matplotlib.pyplot as plt
import math as math
from regression_generation import generate_descent_polynom

STEP = 100
def generateX(func, leftBound, rightBound, step, dim):

    # пока оч долго работает, видимо, надо меньшее количество с большим step=eps генерить
    points = []
    y = []
    i = leftBound

    # короче, генерация кривая немного, в том плане, что на большие размерности сложно обобщать
    # чет такое можно сделать:
    ## while i< rightBound
    ##    x=[zeroes.dim(X)]
    ##    for j in x.size x[j] = random.uniform(leftBound,rightBound) есть нюанс в равномерности, если появятся идеи, как на генерить точек в N-мерном пространстве с нормой
    # расстояния в условный eps, накидайте сюда идей или замутите, не хочется рекурсивно это фигачить
    ##    y append func(x)

    while i <= rightBound:
        x = np.zeros(dim)
        j = leftBound
        while j < rightBound:
            points.append(np.array([i, j]))
            y.append(func([i, j]) + np.random.uniform(-15, 15))
            j += step
        i += step
    return np.array(points), np.array(y)


def generate_sample(batch_size, dim):  # хочется генерить Y вот этой херней, чтобы регулировать размеры батча проще
    x = 0
    while x < batch_size * STEP:
        yield func(x) + np.random.uniform(-1, 1) * np.random.uniform(2, 8)
        x += STEP


def cost_function(A, Y, theta):
    k = Y - A @ theta
    return sum(k ** 2) / len(Y)


def batch_descent(X, Y, lr, startX, EPS, reg, L1_par, L2_par):

    # reg ={0,1,2,3} 0 -> no regularization, 1 -> L1 regularization, 2 -> L2 regularization, 3 -> Elastic(L1+l2)

    theta = np.asarray(startX, dtype='float64')
    previous_cost = math.inf
    current_cost = cost_function(X, Y, theta)
    points = []
    prevtheta = theta + 10
    iter = 0

    dimensions = len(theta)
    batch_size = len(Y)
    while np.abs(current_cost - previous_cost) > EPS:
        prevtheta[1:] = theta[1:]
        previous_cost = current_cost
        derivatives = np.zeros(dimensions)
        # ---------------------------------------------
        for j in range(dimensions):
            summ = 0
            for i in range(batch_size):
                summ += (Y[i] - X[i] @ theta) * X[i][j]
            derivatives[j] = summ / batch_size
        theta += lr * derivatives
        # ---------------------------------------------
        if reg == 0:
            current_cost = cost_function(X, Y, theta)
        else:
            param = 0
            i = 1
            while i < len(theta):
                param = param + theta[i] ** 2
                i = i + 1
            if reg == 1:
                current_cost = cost_function(X, Y, theta) + L1_par * param
            if reg == 2:
                current_cost = cost_function(X, Y, theta) + L2_par * math.sqrt(param)
            if reg == 3:
                current_cost = cost_function(X, Y, theta) + L1_par * (param) + L2_par * math.sqrt(param)

        # print("Batch cost:", current_cost, " i = ", iter, " delta = ",
        #       current_cost - previous_cost, "theta = ", theta)
        iter += 1
        points.append([theta[1], theta[2]])
    return theta, points


def BatchGD(func, leftBnd, rightBnd, startX, EPS, batch_size, lr, reg, L1_par=0.025, L2_par=0.08):
    X, Y = generateX(func, leftBnd, rightBnd, 1, len(startX) - 1)
    A = np.empty((X[:, 1].size, len(startX)))  # 2 - размерность х плюс первый столбец, состоящий из единиц
    A[:, 0] = 1
    A[:, 1:] = X

    print(A)

    theta_batch, points = batch_descent(A, Y, lr, startX, EPS, reg, L1_par, L2_par)

    # короч странная херня с шагом, он какой-то слишком мелкий, при 0.03 уже расходится, при 0.02 через раз
    print(X)
    plt.plot(X, ".")
    # points = np.array(points)
    # plt.plot(points[:, 0], points[:, 1], 'o-')
    # plt.plot(points[-1][0], points[-1][1], 'o-')
    plt.show()

def func(x):
    return 2*(x[0]-4)**2#+3*x[1]**2

startX = [7,-9,9]
EPS = 0.01
lr=0.001
BatchGD(func, -15, 15, startX, EPS, 800, lr, 0, 0.03, 0.08)