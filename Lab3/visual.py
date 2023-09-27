# это файлик для интерфейса
from matplotlib import pyplot as plt


# 1. Визуализация со слоями (комбинация 1 и 2 лаб)

# 2. Я хз крч, это писалось оч давно. Если что-то надумаю, этой надписи тут уже не будет.

def funcToString(init_coefs):
    return "$ Initial: " + " + ".join([
        f"{init_coefs[i]:.3f}" +
        " \cdot x ^{" + str(i) + "}"
        for i in range(len(init_coefs))]) + "$"

def cool_visual(x, yn, func, solver, init_coefs):
    computed = solver.getComputedCoefficients()
    divergence = solver.getDivergence()

    y = func(x, init_coefs)
    plt.figure()
    plt.plot(x, y, label="Initial function", linewidth=2)
    plt.plot(x, yn, label="Randomized data", linewidth=2)
    plt.plot(x, computed, label="Computed")
    plt.plot(x, divergence, label="Divergence", linewidth=2)
    plt.title(funcToString(init_coefs))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()