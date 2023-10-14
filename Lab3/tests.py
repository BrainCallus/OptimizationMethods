import time
import tracemalloc
import numpy as np

from Lab3.lib.absRegression import absRegression

# test_function
def time_test(
        solver: absRegression,
        initX: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        real: np.ndarray = None
) -> float:
    st = time.time_ns()
    solver.recoverCoefs(x, y, initX)
    return (time.time_ns() - st)

# test_function
def result_norm_test(
        solver: absRegression,
        initX: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        real: np.ndarray
) -> float:
    solver.recoverCoefs(x, y, initX)
    computed = solver.getResult()
    return np.linalg.norm(computed - real)

# test_function
def memory_test(
        solver: absRegression,
        initX: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        real: np.ndarray
) -> float:
    tracemalloc.start()
    solver.recoverCoefs(x, y, initX)
    return tracemalloc.get_traced_memory()[1] / 1024
