# это файлик для интерфейса

# 1. Визуализация со слоями (комбинация 1 и 2 лаб)
#


# 2. Я хз крч, это писалось оч давно. Если что-то надумаю, этой надписи тут уже не будет.

def funcToString(init_coefs):
    return "$ Initial: " + " + ".join([
        f"{init_coefs[i]:.3f}" +
        " \cdot x ^{" + str(i) + "}"
        for i in range(len(init_coefs))]) + "$"
