import numpy as np
import model
import cma
from sample import Sample


class DirectSolver(object):
    def __init__(self, _prob, _ntasks, _mean_type):
        self.name = 'CMA-ES'
        self.prob = _prob
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)
        self.model = model.Model(self.prob.dim, self.tasks, _mean_type)
        self.iter_counter = 0
        self.eval_counter = 0
        self.observers = []
        self.iter_values = []
        self.iter_params = []
        print 'model:', self.model

    def add_observer(self, o):
        self.observers += [o]

    def mean(self):
        return self.model.mean

    def solve(self):
        [o.notify_init(self, self.model) for o in self.observers]
        sample_values = []
        for task in self.tasks:
            pt = self.mean().point(task)
            s = Sample(pt, self.prob)
            v = s.evaluate(task)
            sample_values += [v]
        self.iter_values += [sample_values]
        self.iter_params += [self.mean().params()]
        [o.notify_step(self, self.model) for o in self.observers]
        res = {'result': 'NG'}
        # opt = {'verb_time': 0, 'popsize': 16, 'tolfun': 1.0}
        cma.CMAOptions('tol')

        opts = cma.CMAOptions()
        opts.set('verb_disp', 1)
        # opts.set('tolfun', 0.001)
        # opts.set('tolx', 0.0000001)
        # opts.set('tolx', 1.0)
        opts.set('ftarget', 0.001)
        num_offsprings = 16
        opts.set('popsize', num_offsprings)
        max_iter = int(5000 / self.n / num_offsprings)
        print('maxiter: %d' % max_iter)
        opts.set('maxiter', max_iter)
        for key, value in opts.iteritems():
            print '[', key, ']\n', value

        x0 = np.random.rand(self.mean().paramdim) - 0.5

        # print cma.CMAOptions()
        # exit(0)

        print()
        print('------- CMA-ES --------')
        res = cma.fmin(self.evaluate, x0, 1.0, opts)
        print('-----------------------')
        print()
        # np.set_printoptions(precision=6, suppress=True)
        print('the answer: %s' % res[0])

        [o.notify_solve(self, self.model) for o in self.observers]
        return res

    def evaluate(self, _x):
        self.eval_counter += 1
        # Scale up the second part, to cover all the possible range
        x = np.zeros(self.prob.dim * 2)
        for i in range(self.prob.dim):
            x[i] = _x[i]
        for i in range(self.prob.dim, 2 * self.prob.dim):
            x[i] = _x[i] - _x[i - self.prob.dim]
        self.mean().set_params(x)
        sample_values = []
        for task in self.tasks:
            pt = self.mean().point(task)
            s = Sample(pt, self.prob)
            v = s.evaluate(task)
            sample_values += [v]
        avg_error = np.mean(sample_values)
        if np.isnan(avg_error):
            avg_error = 9999.9
        print x, ' value: {', avg_error, '}'
        self.iter_values += [sample_values]  # Values for the entire iterations
        self.iter_params += [x]

        # If one iteration is ended
        if self.eval_counter % 16 == 0:
            self.iter_counter += 1
            # print 'CMA Iteration', self.iter_counter, self.prob.eval_counter
            # for v in self.iter_values:
            #     print sum(v), v
            # print 'best:', self.values()
            sum_values = [sum(vs) for vs in self.iter_values]
            print('sum_values = %s' % sum_values)
            best_index = np.nanargmin(sum_values)
            print('best_index = %d' % best_index)
            best_params = self.iter_params[best_index]
            print('best params = %s' % best_params)
            self.mean().set_params(best_params)
            [o.notify_step(self, self.model) for o in self.observers]

            self.iter_values = []
            self.iter_params = []

        return avg_error

    def num_evals(self):
        return self.prob.eval_counter

    def values(self):
        sum_values = [sum(v) for v in self.iter_values]
        best_index = np.nanargmin(sum_values)
        return self.iter_values[best_index]

    def __str__(self):
        return "[DirectSolver on %s]" % self.prob
