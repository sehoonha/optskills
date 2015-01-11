import numpy as np
from numpy.linalg import norm
import scipy.optimize


class Linear(object):
    def __init__(self, dim, tasks, pts=None):
        self.dim = dim
        self.paramdim = self.dim * 2
        self.tasks = tasks
        self.fit_error = 0.0

        if pts is not None:
            self.fit(pts)
        else:
            # self.a = np.random.rand(dim) - 0.5
            # self.b = np.random.rand(dim) - 0.5
            self.a = np.zeros(dim)
            self.b = np.zeros(dim)

    def set_params(self, params):
        assert(len(params) == self.paramdim)
        self.a = np.array(params[:self.dim])
        self.b = np.array(params[self.dim:])

    def params(self):
        return np.concatenate([self.a, self.b])

    def fit(self, pts, _xdata=None):
        xdata = self.tasks if _xdata is None else _xdata
        a, b = [], []
        self.fit_error = 0

        if self.is_same_pts(pts):
            self.a = np.array(pts[0])
            self.b = np.zeros(self.dim)
            self.fit_error = 0.0
            return

        for i in range(self.dim):
            ydata = [x[i] for x in pts]
            popt, pcov = scipy.optimize.curve_fit(self.fit_func, xdata, ydata)
            a += [popt[0]]
            b += [popt[1]]
            try:
                self.fit_error += sum(np.sqrt(np.diag(pcov)))
            except ValueError:
                print('ValueError: pcov = %s, popt = %s' % (pcov, popt))
        self.a, self.b = np.array(a), np.array(b)

    def is_same_pts(self, pts):
        for i in range(1, len(pts)):
            lhs = pts[0]
            rhs = pts[1]
            if norm(lhs - rhs) > 1e-10:
                return False
        return True

    def fit_func(self, x, a_i, b_i):
        return a_i + x * b_i

    def point(self, w):
        return self.a + w * self.b

    def __str__(self):
        return "{Linear: %s %s (err: %.8f)}" % (self.a, self.b, self.fit_error)


class Cubic(object):
    def __init__(self, dim, tasks, pts=None):
        self.dim = dim
        self.paramdim = self.dim * 4
        self.tasks = tasks
        self.fit_error = 0.0

        if pts is not None:
            self.fit(pts)
        else:
            self.p0 = np.random.rand(dim) - 0.5
            self.p1 = np.random.rand(dim) - 0.5
            self.p2 = np.random.rand(dim) - 0.5
            self.p3 = np.random.rand(dim) - 0.5

    def set_params(self, params):
        assert(len(params) == self.paramdim)
        self.p0 = np.array(params[self.dim * 0: self.dim * 1])
        self.p1 = np.array(params[self.dim * 1: self.dim * 2])
        self.p2 = np.array(params[self.dim * 2: self.dim * 3])
        self.p3 = np.array(params[self.dim * 3: self.dim * 4])

    def params(self):
        return np.concatenate([self.p0, self.p1, self.p2, self.p3])

    def fit(self, pts):
        xdata = self.tasks
        p0, p1, p2, p3 = [], [], [], []
        self.fit_error = 0

        if self.is_same_pts(pts):
            self.p0 = np.array(pts[0])
            self.p1 = np.array(pts[0])
            self.p2 = np.array(pts[0])
            self.p3 = np.array(pts[0])
            self.fit_error = 0.0
            return

        for i in range(self.dim):
            ydata = [x[i] for x in pts]
            popt, pcov = scipy.optimize.curve_fit(self.fit_func, xdata, ydata)
            p0 += [popt[0]]
            p1 += [popt[1]]
            p2 += [popt[2]]
            p3 += [popt[3]]
            try:
                self.fit_error += sum(np.sqrt(np.diag(pcov)))
            except ValueError:
                print('ValueError: pcov = %s' % pcov)
        self.p0, self.p1 = np.array(p0), np.array(p1)
        self.p2, self.p3 = np.array(p2), np.array(p3)

    def is_same_pts(self, pts):
        for i in range(1, len(pts)):
            lhs = pts[0]
            rhs = pts[1]
            if norm(lhs - rhs) > 1e-10:
                return False
        return True

    def fit_func(self, x, p0_i, p1_i, p2_i, p3_i):
        t = x
        t0 = (1 - t) * (1 - t) * (1 - t)
        t1 = 3.0 * (1 - t) * (1 - t) * t
        t2 = 3.0 * (1 - t) * t * t
        t3 = t * t * t
        return t0 * p0_i + t1 * p1_i + t2 * p2_i + t3 * p3_i

    def point(self, w):
        t = w
        t0 = (1 - t) * (1 - t) * (1 - t)
        t1 = 3.0 * (1 - t) * (1 - t) * t
        t2 = 3.0 * (1 - t) * t * t
        t3 = t * t * t
        return t0 * self.p0 + t1 * self.p1 + t2 * self.p2 + t3 * self.p3

    def __str__(self):
        return "{Cubic: %s %s %s %s (err: %.8f)}" % (self.p0, self.p1,
                                                     self.p2, self.p3,
                                                     self.fit_error)


class Interpolation(object):
    def __init__(self, _ntasks):
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)
        pts = []
        pts += [[-0.12613054, 0.32209068, 0.8102112, -0.3177941, -0.0749627]]
        pts += [[0.37903564, 0.18070492, 0.26963919, -0.75517448, 0.13859836]]
        pts += [[-0.08557572, 0.35062343, 0.63688339, 0.68607797, -0.41526881]]
        pts += [[0.39892259, 0.32394091, 0.27865319, -0.2519944, 0.46200161]]
        pts += [[0.2617369, 0.3628393, 0.52119794, -0.24461946, 0.29654157]]
        pts += [[0.36251745, 0.52528973, 0.46150171, -0.60952388, 0.18031131]]

        self.pts = pts
        self.dim = len(self.pts[0])

    def point(self, w):
        pt = np.zeros(self.dim)
        xs = self.tasks
        for i in range(self.dim):
            ys = [p[i] for p in self.pts]
            y = np.interp(w, xs, ys)
            pt[i] = y
        return pt

    def __str__(self):
        return "{Interpolation}"
