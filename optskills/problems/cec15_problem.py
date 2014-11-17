import numpy as np
import cec15


class CEC15(object):
    def __init__(self, _dim, _func_name):
        self.dim = _dim
        self.func_name = _func_name
        self.lo = np.array([-0.5, -0.5])
        self.hi = np.array([0.5, 0.5])
        self.eval_counter = 0  # Well, increasing when simulated

    def center(self, task):
        w = task
        return self.lo * (1 - w) + self.hi * w

    def simulate(self, sample):
        self.eval_counter += 1
        return sample.view(np.ndarray)

    def evaluate(self, result, task):
        c = self.center(task)
        if self.func_name == 'weierstrass':
            return cec15.weierstrass_func(result, Os=c)
        else:
            return 0.0

    def __str__(self):
        return "[CEC15.%s]" % self.func_name
