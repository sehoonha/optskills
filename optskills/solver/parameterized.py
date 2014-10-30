import numpy as np
import model
from sample import Sample


class ParameterizedSolver(object):
    def __init__(self, _prob):
        self.prob = _prob
        self.tasks = np.linspace(0.0, 1.0, 6)
        self.model = model.Model(self.prob.dim, self.tasks, 'linear')
        self.num_parents = 16  # lambda
        self.num_offsprings = 8  # mu
        print 'ParameterizedSolver init OK'

    def solve(self):
        res = {'result': 'NG'}
        MAX_ITER = 1
        for i in range(MAX_ITER):
            self.solve_step(i)
        return res

    def solve_step(self, iteration):
        print('solver iteration: %d' % iteration)

        samples = []
        for i in range(self.num_parents):
            params = self.model.generate_params()
            s = Sample(params, self.prob)
            print("%s %s" % (i, s))
            for task in self.tasks:
                print task, s.evaluate(task)
            samples += [s]

    def __str__(self):
        return "[ParameterizedSolver on %s]" % self.prob
