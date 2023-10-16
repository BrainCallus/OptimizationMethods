import numpy as np

from Lab4.util.grad_util import numeric_grad, pytorch_grad, get_grad
from Lab4.util.run_functions import run_and_return_result

# Тут хочется гитограммку отдельно для времени, отдельно для памяти
def main():
    n = 10
    p = 4
    iters = 500
    rands = [np.random.rand(p) for _ in range(iters)]
    funcs = [lambda x: sum(sum(3 * rands[_][i] * x ** i for i in range(p))) for _ in range(iters)]
    xs = [np.random.rand(n) * 3 for _ in range(iters)]

    deviations = [0, 0, 0]
    time_usage = [0, 0, 0]
    memory_usage = [0, 0, 0]
    for f, x in zip(funcs, xs):
        result = run_and_return_result(lambda: numeric_grad(f, x))
        ref_result = result.result
        time_usage[0], memory_usage[0] = time_usage[0] + result.time_usage, memory_usage[0] + result.memory_usage

        result = run_and_return_result(lambda: get_grad(f, x))
        deviations[1] = deviations[1] + np.linalg.norm(ref_result - result.result)
        time_usage[1], memory_usage[1] = time_usage[1] + result.time_usage, memory_usage[1] + result.memory_usage

        result = run_and_return_result(lambda: pytorch_grad(f, x))
        deviations[2] = deviations[2] + np.linalg.norm(ref_result - result.result)
        time_usage[2], memory_usage[2] = time_usage[2] + result.time_usage, memory_usage[2] + result.memory_usage

    deviations, time_usage, memory_usage = map(lambda l: '\t'.join(map(lambda x: str(x / iters), l)),
                                               [deviations, time_usage, memory_usage])
    print(
        f"Methods: \tmanual\tnumdifftools\tpytorch\nDeviations:\t{deviations}\nTimes\t{time_usage}\nMems:\t{memory_usage}")


if __name__ == "__main__":
    main()
