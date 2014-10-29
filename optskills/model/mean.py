import numpy as np


class Linear(object):
    def __init__(self, dim, tasks, pts=None):
        self.dim = dim
        self.tasks = tasks

        if pts is not None:
            self.fit(pts)
        else:
            self.a = np.zeros(dim)
            self.b = np.ones(dim)

    def fit(self, pts):
        pass

    def point(self, w):
        return self.a + w * self.b

    def __str__(self):
        return "{Linear: %s %s}" % (self.a, self.b)
