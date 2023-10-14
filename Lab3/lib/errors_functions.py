def quadratic_error_func(x, func, data):
    return sum([(data[i][1] - func(data[i][0], x)) ** 2 for i in range(len(data))])
