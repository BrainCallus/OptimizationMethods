from kind_of_interface import *
import math


def main():

    # analyze_random_function(800, 750)
    draw_random_function(4)

    #multiple_tests_to_excel() # осторожно, работает очень-очень долго
    # + необходимо заранее проверить, что существует папка ./tables


    # analyse_eps_parameter(used_func1, used_grad1)
    # analyse_eps_parameter(used_func2, used_grad2)
    # analyse_eps_parameter(used_func3, used_grad3)

    # func_set = func_set_3
    # scaling_vector = [1 / math.sqrt(4.5), 1 / math.sqrt(2)]
    # method = 1
    #
    # func1 = func_set[0]
    # grad1 = func_set[1]
    # func_name = func_set[-1]
    #
    # func2 = scale_function(2, func1, scaling_vector)
    # grad2 = scale_function(2, grad1, scaling_vector)
    #
    # functions_compare(func1, grad1, func2, grad2, method, func_name)

if __name__ == "__main__":
    main()
