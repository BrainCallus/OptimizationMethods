from kind_of_interface import *


def main():

    # n = 100
    #
    # for i in range(n, n + 1):
    #     for j in range(n, n + 1):
    #         print("перем  (n): " + str(i))
    #         print("собств (k): " + str(j))
    #         dimensions = i
    #         self_number = j
    #         func, grad, matrix = generate_random_function(dimensions, self_number)
    #         analyze_function(dimensions, func, grad)

    dimensions = 2
    self_number = 3
    func, grad, a = generate_random_function_quadratic(dimensions, self_number)
    draw_function_graph(a, func, grad)


    # a = [[1, -1], [0, 2]]
    # func = function_generator(dimensions, a)
    # grad = gradient_generator(dimensions, a)
    # draw_function_graph(a, func, grad, 0.1)
    #
    # a = [[4, 0], [0, 0.5]]
    # func = function_generator(dimensions, a)
    # grad = gradient_generator(dimensions, a)
    # draw_function_graph(a, func, grad, 0.1)



    # for i in range(1, 1001, 100):
    #     for j in range(2, 1002, 100):
    #         dimensions = i
    #         self_number = j
    #         func, grad, vector = generate_random_function_vector(dimensions, self_number)
    #
    #         print('Перем : ' + str(dimensions))
    #         print('Собств: ' + str(self_number))
    #         analyze_function(dimensions, func, grad, vector)


    # analyze_function(2, lambda a: a[0]**2 + 302 * a[1]**2, lambda a: np.asarray([a[0]*2, 302 * a[1]*2]), [1, 302])

if __name__ == "__main__":
    main()
