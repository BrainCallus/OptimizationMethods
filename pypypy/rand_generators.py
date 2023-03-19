import numpy as np
from random import uniform
from random import randint
from random import choice


def generate_diagonal_1(k, n, given_vector=None):
    gen_range = 1000 / k
    if given_vector is None:
        given_vector = np.zeros((n,), dtype=float)
    min_gen = uniform(0, gen_range)
    max_gen = min_gen * k
    for i in range(n):
        given_vector[i] = uniform(min_gen, max_gen)
    a = randint(0, n - 1)
    given_vector[a] = min_gen
    if n > 1:
        b = choice([i for i in range(n) if i != a])
        given_vector[b] = max_gen
    return given_vector


def matrix_out_of_vector(n, given_vector, given_matrix=None):
    if given_matrix is None:
        given_matrix = np.zeros(shape=(n, n))
    for i in range(n):
        given_matrix[i][i] = given_vector[i]
    return given_matrix

