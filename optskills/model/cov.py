import numpy as np


class Cov(object):
    def __init__(self, dim, center, pts=None, _C=None):
        self.dim = dim
        self.m = center

        if pts is not None:
            self.fit(pts)
        elif _C is not None:
            self.C = _C
        else:
            self.C = np.identity(dim)

    def fit(self, pts):
        """each row of pts is a point"""
        # np.matrix subtracted by np.array (for every row)
        X = pts - self.m
        self.C = np.cov(X, rowvar=0)  # A row is an observation, not variable

    def interploate(self, rhs):
        # lhs = self
        pass

    def generate_params(self, stepsize=1.0):
        mean = self.m
        cov = self.C * stepsize
        params = np.random.multivariate_normal(mean, cov)
        return params

    def __str__(self):
        return "{Cov %s diag:%s}" % (self.m, self.C.diagonal())
