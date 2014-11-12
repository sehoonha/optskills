import numpy as np
from numpy.linalg import norm


class MirroredSphere(object):
    def __init__(self):
        self.dim = 2
        # self.lo = [np.array([-0.5, -0.25]), np.array([-0.5, 0.25])]
        # self.hi = [np.array([0.5, -0.25]), np.array([0.5, 0.25])]

        # self.lo = [np.array([-0.5, -0.5]), np.array([-0.5, 0.5])]
        # self.hi = [np.array([0.5, -0.5]), np.array([0.5, 0.5])]

        self.lo = [np.array([-0.5, -0.75]), np.array([-0.5, 0.75])]
        self.hi = [np.array([0.5, -0.75]), np.array([0.5, 0.75])]

        self.eval_counter = 0  # Well, increasing when simulated

    def center(self, task, segment):
        i = segment
        w = task
        return self.lo[i] * (1 - w) + self.hi[i] * w

    def simulate(self, sample):
        self.eval_counter += 1
        return sample.view(np.ndarray)

    def evaluate(self, result, task):
        segment = 0 if result[1] < 0.0 else 1
        c = self.center(task, segment)
        return norm(c - result)

    def __str__(self):
        return "[MirroredSphereProblem]"
