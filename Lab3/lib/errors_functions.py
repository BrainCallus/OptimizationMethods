def error(x, func, param):
    return param[1] - func(param[0], x)


def quadratic_error_func(x, func, data):
    return sum([error(x, func, data[i]) ** 2 for i in range(len(data))])
