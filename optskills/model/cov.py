import numpy as np


class Cov(object):
    def __init__(self, dim, center, pts=None):
        self.dim = dim
        self.center = center

        if pts is not None:
            self.fit(pts)
        else:
            self.C = np.identity(dim)

    def fit(self, pts):
        pass

    def __str__(self):
        return "{Cov %s diag:%s}" % (self.center, self.C.diagonal())
