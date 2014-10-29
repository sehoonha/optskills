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

    def func(self, w, *params):
        pass

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
