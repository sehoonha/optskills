import numpy as np
import scipy.interpolate
import cec15
import matplotlib.pyplot as plt


class CEC15(object):
    def __init__(self, _dim, _func_name, _pts=None, _seg_type='linear',
                 _fscale=1.0):
        self.dim = _dim
        self.func_name = _func_name

        if _pts is None:
            self.pts = []
            self.pts += [np.array([-0.5] * self.dim)]
            self.pts += [np.array([0.5] * self.dim)]
        else:
            self.pts = _pts
        self.seg_type = _seg_type
        self.fscale = _fscale
        # self.plot_segment()
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
        # return self.lo * (1 - w) + self.hi * w

    def plot_segment(self):
        tasks = np.linspace(0.0, 1.0, 11)
        pts = [self.center(t) for t in tasks]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]

        fig = plt.figure()
        fig.set_size_inches(12.0, 12.0)
        plt.plot(x, y)
        plt.axes().set_aspect('equal', 'datalim')
        plt.axes().set_xlim(-1.0, 1.0)
        plt.axes().set_ylim(-1.0, 1.0)
        plt.savefig('plot_segment.png')
        exit(0)

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
        return f * self.fscale + 0.1 * penalty

    def __str__(self):
        return "[CEC15.%s]" % self.func_name
