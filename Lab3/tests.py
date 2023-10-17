import time
import tracemalloc
import numpy as np

from Lab3.lib.absRegression import absRegression

# test_func
class time_test:

        ax_name = "milliseconds"

        test_name = "time test"
    
        def exec(
                solver: absRegression,
                initX: np.ndarray,
                x: np.ndarray,
                y: np.ndarray,
                real: np.ndarray = None
        ) -> float:
                st = time.time_ns()
                solver.recoverCoefs(x, y, initX)
                return (time.time_ns() - st)


# test_func
class result_norm_test:

        ax_name = "difference norm"

        test_name = "difference norm test"
    
        def exec(
                solver: absRegression,
                initX: np.ndarray,
                x: np.ndarray,
                y: np.ndarray,
                real: np.ndarray = None
        ) -> float:
                solver.recoverCoefs(x, y, initX)
                computed = solver.getResult()
                return np.linalg.norm(computed - real)
        

# test_func
class memory_test:

        ax_name = "bytes"

        test_name = "memory test"
    
        def exec(
                solver: absRegression,
                initX: np.ndarray,
                x: np.ndarray,
                y: np.ndarray,
                real: np.ndarray = None
        ) -> float:
                tracemalloc.start()
                solver.recoverCoefs(x, y, initX)
                a = tracemalloc.get_traced_memory()[1] / 1024
                tracemalloc.stop()
                return a



# test_func
class iter_test:

        ax_name = "iterations number"

        test_name = "iterations test"
    
        def exec(
                solver: absRegression,
                initX: np.ndarray,
                x: np.ndarray,
                y: np.ndarray,
                real: np.ndarray = None
        ) -> float:
                _, iter = solver.recoverCoefs(x, y, initX)
                return iter
