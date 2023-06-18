from OptimizationMethods.Lab3.abs import absNewton


class GN_Met(absNewton):
    def getSigma(self, diverg):
        sub_iters, jacobian = self.compute_jacobian(self.coefficients, step=10 ** (-3))
        return sub_iters, self.pseudo_inverse(jacobian) @ diverg
