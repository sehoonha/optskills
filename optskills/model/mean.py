import numpy as np
from numpy.linalg import norm
import scipy.optimize


class Linear(object):
    def __init__(self, dim, tasks, pts=None):
        self.dim = dim
        self.paramdim = self.dim * 2
        self.tasks = tasks

        if pts is not None:
            self.fit(pts)
        else:
            self.a = np.random.rand(dim) - 0.5
            self.b = np.random.rand(dim) - 0.5

    def fit(self, pts):
        xdata = self.tasks
        a, b = [], []
        for i in range(self.dim):
            ydata = [x[i] for x in pts]
            popt, pcov = scipy.optimize.curve_fit(self.fit_func, xdata, ydata)
            a += [popt[0]]
            b += [popt[1]]
        self.a, self.b = np.array(a), np.array(b)

    def fit_func(self, x, a_i, b_i):
        return a_i + x * b_i

    def point(self, w):
        return self.a + w * self.b

    def __str__(self):
        return "{Linear: %s %s}" % (self.a, self.b)


# from scipy.optimize import curve_fit


# def f(x, *params):
#     print params
#     (a, b, c) = params
#     return a * np.exp(-b * x) + c

# xdata = np.linspace(0, 4, 50)
# y = f(xdata, 2.5, 1.3, 0.5)
# ydata = y + 0.2 * np.random.normal(size=len(xdata))
# popt, pcov = curve_fit(f, xdata, ydata, np.random.rand(3))
# print popt
# print pcov
