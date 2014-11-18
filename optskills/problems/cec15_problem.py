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
        penalty = 0.0
        for i in range(self.dim):
            if result[i] > 1.0:
                penalty += (result[i] - 1.0) ** 2
            if result[i] < -1.0:
                penalty += (result[i] - (-1.0)) ** 2

        f = 0.0
        if self.func_name == 'bent_cigar':
            f = cec15.bent_cigar_func(result, Os=c)
        elif self.func_name == 'weierstrass':
            f = cec15.weierstrass_func(result, Os=c)
        elif self.func_name == 'schwefel':
            f = cec15.schwefel_func(result, Os=c)
        else:
            f = 0.0
        return f + 0.1 * penalty

    def __str__(self):
        return "[CEC15.%s]" % self.func_name
