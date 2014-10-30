import numpy as np
from numpy.linalg import norm


class Sphere(object):
    def __init__(self):
        self.dim = 2
        self.lo = np.array([-0.5, -0.5])
        self.hi = np.array([1.0, 1.0])

    def center(self, task):
        w = task
        return self.lo * (1 - w) + self.hi * w

    def simulate(self, sample):
        return sample.view(np.ndarray)

    def evaluate(self, result, task):
        c = self.center(task)
        return norm(c - result)

    def __str__(self):
        return "[SphereProblem]"
