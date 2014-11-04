import numpy as np
import model
import cma
from sample import Sample


class DirectSolver(object):
    def __init__(self, _prob, _ntasks, _mean_type):
        self.prob = _prob
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)
        self.model = model.Model(self.prob.dim, self.tasks, _mean_type)
        self.observers = []
        print 'model:', self.model

    def add_observer(self, o):
        self.observers += [o]

    def mean(self):
        return self.model.mean

    def solve(self):
        res = {'result': 'NG'}
        opt = {'verb_time': 0}
        x0 = np.random.rand(self.mean().paramdim) - 0.5

        print
        print '------- CMA-ES --------'
        res = cma.fmin(self.evaluate, x0, 2.0, opt)
        print '-----------------------'
        print
        # np.set_printoptions(precision=6, suppress=True)
        print 'the answer:', res[0]

        [o.notify_solve(self, self.model) for o in self.observers]
        return res

    def evaluate(self, x):
        self.mean().set_params(x)
        sum_error = 0.0
        for task in self.tasks:
            pt = self.mean().point(task)
            s = Sample(pt, self.prob)
            sum_error += s.evaluate(task)
        return sum_error

    def __str__(self):
        return "[DirectSolver on %s]" % self.prob
