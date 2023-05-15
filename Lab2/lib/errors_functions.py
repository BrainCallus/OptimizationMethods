def error(x, dot):
    res = 0
    for i in range(len(x)):
        res += x[i] * dot[0] ** i
    return res - dot[1]

def quadratic_error_func(x, data):
    err = [(error(x, data[i]) ** 2) for i in range(len(data))]
    return sum(err)

def quadratic_error_func_grad(x, dot):
    a = error(x, dot) * 2
    return [a * dot[0] ** i for i in range(len(x))]