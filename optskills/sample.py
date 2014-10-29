import numpy as np


class Sample(np.ndarray):
    def __new__(subtype, buffer, prob=None, shape=None, dtype=float, offset=0,
                strides=None, order=None):
        buffer = np.array([float(x) for x in buffer])
        if shape is None:
            shape = (len(buffer),)
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                 strides, order)
        obj.prob = prob
        obj.result = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.prob = getattr(obj, 'prob', None)
        self.result = None
        # We do not need to return anything

    def simulate(self):
        self.result = self.prob.simulate(self)

    def is_simulated(self):
        return self.result is not None

    def evaluate(self, task):
        if not self.is_simulated():
            self.simulate()
        return self.prob.evaluate(self.result, task)

    def __str__(self):
        return "S[" + ", ".join(["%.6f" % x for x in self]) + "]"

    def __repr__(self):
        ret = ""
        ret += "Sample("
        ret += ", ".join(["%.8f" % x for x in self])
        ret += ", " + repr(self.prob) + ")"
        return ret
