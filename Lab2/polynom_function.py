def polynom(vector):
    def func(x):
        r = [vector[i] * x ** i for i in vector]
        return sum(r)
    def grad(x):
        r = [i * vector[i] * x ** (i - 1) for i in vector]
        return sum(r)
    return func, grad