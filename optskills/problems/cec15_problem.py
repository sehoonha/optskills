import numpy as np
import scipy.interpolate
import cec15
import matplotlib.pyplot as plt


class CEC15(object):
    def __init__(self, _dim, _func_name, _pts=None, _seg_type='linear',
                 _fscale=1.0, _adjust=None):
        self.dim = _dim
        self.func_name = _func_name

        if _pts is None:
            self.pts = []
            self.pts += [np.array([-0.5] * self.dim)]
            self.pts += [np.array([0.5] * self.dim)]
        else:
            self.pts = _pts

        # Allow function type shortcut
        if _seg_type == 'quad':
            self.seg_type = 'quadratic'
        else:
            self.seg_type = _seg_type

        self.fscale = _fscale

        if _adjust is None:
            self.adjust = [1.0, 1.0]
        else:
            self.adjust = _adjust

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

    def adjust_param(self, task):
        n = len(self.adjust)
        w = task
        x = np.linspace(0.0, 1.0, n)
        y = self.adjust
        f = scipy.interpolate.interp1d(x, y)
        return f(w)

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

        # Additional debug routine
        for t in tasks:
            print 'Task:', t, ' Adjust:', self.adjust_param(t)
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

        ap = self.adjust_param(task)
        sr = self.scale_rate(task)

        f = 0.0
        if self.func_name == 'bent_cigar':
            f = cec15.bent_cigar_func(result, Os=c, adjust=ap)
        elif self.func_name == 'weierstrass':
            f = cec15.weierstrass_func(result, Os=c, sr=sr)
        elif self.func_name == 'schwefel':
            f = cec15.schwefel_func(result, Os=c)
        else:
            f = 0.0
        return f * self.fscale + 0.1 * penalty

    def scale_rate(self, task):
        if self.func_name == 'weierstrass':
            x = [0.0, 1.0]
            y = [0.2, 2.0]
            f = scipy.interpolate.interp1d(x, y)
            return f(task)
        return None

    def __str__(self):
        return "[CEC15.%s]" % self.func_name
