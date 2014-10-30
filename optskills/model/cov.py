import numpy as np


class Cov(object):
    def __init__(self, dim, center, pts=None):
        self.dim = dim
        self.m = center

        if pts is not None:
            self.fit(pts)
        else:
            self.C = np.identity(dim)

    def fit(self, pts):
        n = len(pts)
        dim = self.dim
        C = np.zeros(dim, dim)
        for x in pts:
            d = x - self.m  # Difference from the center
            C += d.reshape(dim, 1) * d
        # should we specially care when n = 1?
        C /= (n - 1)
        self.C = C

    def interploate(self, rhs):
        # lhs = self
        pass

    def generate_params(self):
        mean = self.m
        cov = self.C
        params = np.random.multivariate_normal(mean, cov)
        return params

    def __str__(self):
        return "{Cov %s diag:%s}" % (self.m, self.C.diagonal())
