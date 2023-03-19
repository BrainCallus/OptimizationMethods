def function_out(n, given_matrix):
    for i in range(n):
        for j in range(n - i):
            if j != 0:
                print("(" + str(given_matrix[i][i + j]) + ") * x_{" + str(i) + "} * x_{" + str(i + j) + "} +", end=" ")
            else:
                print("(" + str(given_matrix[i][i + j]) + ") * x_{" + str(i) + "}^2", end=" ")
                if i != n - 1:
                    print("+ ")
                else:
                    print("\n")
    print()

# вывод нового способа задания через x_{i}^2


def quadratic_function_out(n, given_vector):
    for i in range(n - 1):
        print("(" + str(given_vector[i]) + ") * x_{" + str(i), end="}^2 + ")
    print("(" + str(given_vector[n - 1]) + ") * x_{" + str(n - 1) + "}^2")
    print()


# для функций двух переменных
# (такой вывод можно будет сразу бабахнуть, к примеру, в Desmos)

def function_two_out(given_matrix):
    s = '$f (x, y) = '
    flag = False
    if given_matrix[0][0] != 0:
        flag = True
        s += str(round(given_matrix[0][0], 3)) + " \cdot x^2 "
    if given_matrix[0][1] + given_matrix[0][1] != 0:
        if given_matrix[0][1] + given_matrix[0][1] > 0 and flag:
            s += " + " + str(round(given_matrix[0][1] + given_matrix[0][1], 3)) + " \cdot xy "
        else:
            s += str(round(given_matrix[0][1] + given_matrix[0][1], 3)) + " \cdot xy "
        flag = True
    if given_matrix[1][1] != 0:
        if given_matrix[1][1] > 0 and flag:
            s += " + " + str(round(given_matrix[1][1], 3)) + " \cdot y^2 "
        else:
            s += str(-(round(given_matrix[1][1], 3))) + " \cdot y^2 "
    s += '$'
    return s


# просто вывод списка для удобства

def print_array(n, given_matrix):
    for i in range(n):
        print(given_matrix[i])
    print()

