from pypypy.rand_generators import *

def function_generator(n, given_matrix, given_vector=None, given_monom=0):
    flag = False if given_vector is None else True

    def generated_function(vector):
        answer = given_monom
        for i in range(n):
            if flag:
                answer += vector[i] * given_vector[i]
            for j in range(n):
                answer += vector[i] * vector[j] * given_matrix[i][j]
        return answer

    return generated_function


def gradient_generator(n, given_matrix, given_vector=None):
    flag = False if given_vector is None else True

    def generated_gradient(vector):
        answer = np.zeros((n,), dtype='float64')
        for i in range(n):
            if flag:
                answer[i] += given_vector[i]
            for j in range(n):
                answer[j] += vector[i] * given_matrix[i][j]
                answer[i] += vector[j] * given_matrix[i][j]
        return answer

    return generated_gradient


def function_generator_vector(n, given_vector):

    def generated_function(vector):
        answer = 0
        for i in range(n):
            answer += given_vector[i] * (vector[i] ** 2)
        return answer

    return generated_function


def gradient_generator_vector(n, given_vector):

    def generated_gradient(vector):
        answer = np.zeros((n,), dtype='float64')
        for i in range(n):
            answer[i] += 2 * vector[i] * given_vector[i]
        return answer

    return generated_gradient