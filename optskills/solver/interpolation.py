import numpy as np
import model
import cma
from sample import Sample


class InterpolationSolver(object):
    def __init__(self, _prob, _ntasks, _mean_type):
        self.name = 'Interpolation'
        self.prob = _prob
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)
        self.current_task = None
        self.model = model.Model(self.prob.dim, self.tasks, _mean_type)
        self.iter_counter = 0
        self.eval_counter = 0
        self.observers = []
        self.iter_values = []
        print 'model:', self.model

    def add_observer(self, o):
        self.observers += [o]

    def mean(self):
        return self.model.mean

    def solve(self):
        [o.notify_init(self, self.model) for o in self.observers]
        res = {'result': 'NG'}

        # 1. The initial step notification
        [o.notify_step(self, self.model) for o in self.observers]

        print('Solving...')
        pts = []
        for task in self.tasks:
            task = 0.4  # Test
            print('')
            print('------- CMA-ES : task = %.6f -------- ' % task)

            self.current_task = task
            opts = cma.CMAOptions()
            opts = cma.CMAOptions()
            opts.set('verb_disp', 1)
            opts.set('ftarget', 0.001)
            opts.set('popsize', 16)
            opts.set('maxiter', 300)

            # opt = {'verb_time': 0, 'popsize': 16, 'tolfun': 1e-5}
            x0 = np.random.rand(self.prob.dim) - 0.5
            res = cma.fmin(self.evaluate, x0, 2.0, opts)
            print('------------ task = %.6f ----------- ' % task)
            print('self.eval_counter = %d' % self.num_evals())
            print('the answer: %s' % res[0])
            pts += [res[0]]
            print('')

            # [o.notify_step(self, self.model) for o in self.observers]
        self.mean().fit(pts)

        # 2. The final step notification
        [o.notify_step(self, self.model) for o in self.observers]
        for i in range(self.n):
            print("%d (%.4f) : %s" % (i, self.tasks[i], pts[i]))
        print('sum values: %.6f' % np.mean(self.values()))
        print self.model

        [o.notify_solve(self, self.model) for o in self.observers]
        return res

    def evaluate(self, x):
        self.eval_counter += 1
        s = Sample(x, self.prob)
        v = s.evaluate(self.current_task)
        # print('%.6f <--- %s' % (v, x))
        return v

    def num_evals(self):
        return self.prob.eval_counter

    def values(self):
        sample_values = []
        for task in self.tasks:
            pt = self.mean().point(task)
            s = Sample(pt, self.prob)
            v = s.evaluate(task)
            sample_values += [v]
        return sample_values

    def __str__(self):
        return "[InterpolationSolver on %s]" % self.prob
