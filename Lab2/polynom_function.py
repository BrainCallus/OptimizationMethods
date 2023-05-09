def polynom(vector):
    def res(x):
        r = [vector[i] * x ** i for i in range(len(vector))]
        return sum(r)
    return res