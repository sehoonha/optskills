import numpy as np


class Sample(np.ndarray):
    def __new__(subtype, buffer, prob, shape=None, dtype=float, offset=0,
                strides=None, order=None):
        buffer = np.array([float(x) for x in buffer])
        print 'buffer:', buffer
        if shape is None:
            shape = (len(buffer),)
        print 'shape:', shape
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                 strides, order)
        obj.prob = prob
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.prob = getattr(obj, 'prob', None)
        # We do not need to return anything

    def __str__(self):
        return "[" + ", ".join(["%.6f" % x for x in self]) + self.prob + "]"
