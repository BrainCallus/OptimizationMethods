import math


from Lab3.lib.absRegression import absRegression
from Lab3.lib.errors_functions import quadratic_error_func
from Lab3.lib.functions import *


class absBFGS(absRegression, ABC):
    def __init__(self,
                 eps: float = 10 ** (-3)):
        super().__init__(eps=eps)
        self.gradient = None
        self.func = None
        self.coefficients = None
        self.y = None
        self.x = None
        self.type = MiniBatchGD
        self.type_args = lambda: [80]

    def recoverCoefs(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     init_data: np.ndarray):
        self.x = x
        self.y = y
        self.coefficients = init_data
        self.func = self.type(lambda a, b: quadratic_error_func(a, self.function, b),
                              lambda a, b: self.grad(quadratic_error_func, a, self.function, b),
                              np.dstack((x, y))[0], *self.type_args())
        i = self.execute()
        return i, i

    def line_search(self, x, p, c1=0.001, c2=0.9, alf=1):
        nabl = self.func.get_grad(x)
        fx = self.func.func(x)
        new_x = x + alf * p
        new_nabl = self.func.get_grad(new_x)
        f_new_x = self.func.func(new_x)
        while ((f_new_x > fx + (c1 * alf * nabl.T @ p)
                or math.fabs(new_nabl.T @ p) > c2 * math.fabs(nabl.T @ p)) and abs(f_new_x - fx) >= 1e-8):
            alf *= 0.5
            new_x = x + alf * p
            f_new_x = self.func.func(new_x)
            new_nabl = self.func.get_grad(new_x)
        return alf

    @abstractmethod
    def execute(self):
        ...


class BFGS(absBFGS):
    def __init__(self, max_iter: int = 100):
        self.type = MiniBatchGD
        self.type_args = lambda: [80]
        self.eps = 1e-3
        self.max_iter = max_iter

    def execute(self):
        i = 1
        dim = len(self.coefficients)
        H = np.eye(dim)
        I = np.eye(dim)
        nabl = (self.func.get_grad(self.coefficients))
        c = 1e-7
        delta = 100
        while delta > self.eps and i < self.max_iter:
            p = -H @ nabl
            alf = self.line_search(self.coefficients, p)
            s = alf * p
            self.coefficients += s
            new_nabl = self.func.get_grad(self.coefficients)
            y = new_nabl - nabl
            p = y @ s
            ro = 1.0 / (p + c)
            A1 = I - ro * s[:, np.newaxis] * y[np.newaxis, :]
            A2 = I - ro * y[:, np.newaxis] * s[np.newaxis, :]
            H = A1 @ (H @ A2) + (ro * s[:, np.newaxis] * s[np.newaxis, :])
            delta = np.linalg.norm(nabl - new_nabl)
            nabl = new_nabl
            i += 1
        print(self.coefficients)
        return i


class L_BFGS(absBFGS):
    def __init__(self, max_iter: int = 100, queue_size: int = 15):
        self.queue_sz = queue_size
        self.type = MiniBatchGD
        self.type_args = lambda: [50]
        self.eps = 1e-4
        self.max_iter = max_iter

    def execute(self):
        main_list = LinkedList(self.queue_sz)
        i = 1
        grad = self.func.get_grad(self.coefficients)
        x_prev = np.zeros(len(self.coefficients))
        grad_prev = grad - grad
        ys = 100
        c = 1e-9

        while np.linalg.norm(ys) > self.eps and i < self.max_iter:
            q = self.func.get_grad(grad)

            nnode = main_list.last
            while nnode is not None:
                al = (np.dot(nnode.s, q) * nnode.rho)
                nnode.alpha = al
                q -= al * nnode.y
                nnode = nnode.prev

            gamma = 1.0
            if main_list.size != 0:
                nnode = main_list.first
                ys = nnode.y
                gamma = np.dot(nnode.y, nnode.s) / (np.dot(nnode.y, nnode.y) + c)
            r = q * gamma

            nnode = main_list.first
            while nnode is not None:
                r += nnode.s * (nnode.alpha - nnode.rho * np.dot(nnode.y, r))
                nnode = nnode.next

            alf = self.line_search(self.coefficients, -r)
            self.coefficients -= r * alf
            grad = self.func.get_grad(self.coefficients)

            if main_list.size != 0:
                main_list.insert(self.coefficients - x_prev, grad - grad_prev,
                                 1.0 / (np.dot(main_list.first.y, main_list.first.s) + c),
                                 alf)
            else:
                main_list.insert(self.coefficients - x_prev, grad - grad_prev, 1e-3, alf)

            x_prev = self.coefficients
            grad_prev = grad
            i += 1
        # print(i, ys, main_list.first.s, self.coefficients, xPrev)
        print(self.coefficients)
        return i


class Node:
    def __init__(self, s, y, rho, alpha):
        self.s = s
        self.y = y
        self.rho = rho
        self.alpha = alpha
        self.next = None
        self.prev = None


class LinkedList:
    def __init__(self, queue_sz):
        self.max_size = queue_sz
        self.first = None
        self.last = None
        self.size = 0

    def insert(self, s, y, rho, alpha):
        n = Node(s, y, rho, alpha)
        if self.size == 0:
            self.size = 1
            self.first = n
            self.last = n
        elif self.size == 1:
            self.size = 2
            self.first = n
            self.first.next = self.last
            self.last.prev = self.first
        elif self.size != self.max_size:
            self.size += 1
            self.first.prev = n
            n.next = self.first
            self.first = n
        else:
            self.first.prev = n
            n.next = self.first
            self.first = n
            self.last = self.last.prev
            self.last.next = None
