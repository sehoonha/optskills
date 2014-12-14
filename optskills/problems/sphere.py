import numpy as np
from numpy.linalg import norm
import scipy.interpolate


class Sphere(object):
    def __init__(self, _dim=2, _seg_type='linear'):
        self.dim = _dim

        self.seg_type = _seg_type
        if self.seg_type == 'linear':
            self.pts = []
            self.pts += [np.array([-0.5] * self.dim)]
            self.pts += [np.array([0.5] * self.dim)]
        elif self.seg_type == 'cubic':
            self.pts = []
            self.pts += [np.array([-0.5, -0.5])]
            self.pts += [np.array([-0.5, 0.0])]
            self.pts += [np.array([0.0, 0.0])]
            self.pts += [np.array([0.0, 0.5])]

        self.eval_counter = 0  # Well, increasing when simulated

    def center(self, task):
        n = len(self.pts)
        w = task
        x = np.linspace(0.0, 1.0, n)
        center = [0.0] * self.dim
        for i in range(self.dim):
            y = [p[i] for p in self.pts]
            f = scipy.interpolate.interp1d(x, y, self.seg_type)
            center[i] = f(w)
        return np.array(center)

    def simulate(self, sample):
        self.eval_counter += 1
        return sample.view(np.ndarray)

    def evaluate(self, result, task):
        c = self.center(task)
        return norm(c - result)

    def __str__(self):
        return "[SphereProblem (%s)]" % self.seg_type
