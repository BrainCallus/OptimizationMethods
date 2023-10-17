import numpy as np

from Lab4.util.grad_util import numeric_grad, pytorch_grad, get_grad
from Lab4.util.graphic_util import draw_hist
from Lab4.util.run_functions import run_and_return_result, RunResult


def main():
    n = 20
    p = 10
    iters = 500
    rands = [np.random.rand(p) for _ in range(iters)]
    funcs = [lambda x: sum(sum(3 * rands[_][i] * x ** i for i in range(p))) for _ in range(iters)]
    xs = [np.random.rand(n) * 3 for _ in range(iters)]

    grads = [
        (numeric_grad, 'our implementation'),
        (get_grad, 'numdifftools'),
        (pytorch_grad, 'pytorch')
    ]
    named_results = {'our implementation': RunResult(0, 0, 0), 'numdifftools': RunResult(0, 0, 0),
                     'pytorch': RunResult(0, 0, 0)}
    for f, x in zip(funcs, xs):
        ref_result = 0
        for method, name in grads:
            result = run_and_return_result(lambda: method(f, x))
            if name == 'our implementation':
                ref_result = result.result
                named_results.get(name).add(result)
            else:
                named_results.get(name).add(
                    RunResult(np.linalg.norm(ref_result - result.result), result.time_usage, result.memory_usage))

    named_results = list(named_results.items())

    draw_hist(named_results, lambda x: x[1].result / iters, 'Different differencing method derivations')
    draw_hist(named_results, lambda x: x[1].time_usage / iters, 'Different differencing method time usage')
    draw_hist(named_results, lambda x: x[1].memory_usage / iters, 'Different differencing method memory_usage')


if __name__ == "__main__":
    main()
