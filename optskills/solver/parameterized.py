import numpy as np
import model
from sample import Sample


class ParameterizedSolver(object):
    def __init__(self, _prob):
        self.prob = _prob
        self.n = 6
        self.tasks = np.linspace(0.0, 1.0, self.n)
        self.model = model.Model(self.prob.dim, self.tasks, 'linear')
        self.num_parents = 16  # lambda
        self.num_offsprings = 4  # mu
        print 'ParameterizedSolver init OK'

    def solve(self):
        res = {'result': 'NG'}
        MAX_ITER = 10
        best_samples = [[] for i in range(self.n)]
        for i in range(MAX_ITER):
            next_best_samples = self.solve_step(i, best_samples)
            best_samples = next_best_samples
        return res

    def solve_step(self, iteration, best_samples):
        print('solver iteration: %d' % iteration)
        print('best samples: %s' % best_samples)

        samples = []
        # Generate all samples
        for i in range(self.num_parents):
            # Generate params from the model and make a sample
            params = self.model.generate_params()
            s = Sample(params, self.prob)
            s.iteration = iteration
            s.simulate()
            samples += [s]
            j = self.model.debug_last_generate_index
            print("%s (from %d) %s" % (i, j, s))

        # Select samples based on the criteria
        selected = []
        next_best_samples = []
        for task, best in zip(self.tasks, best_samples):
            key = lambda s: s.evaluate(task)
            sorted_samples = sorted(samples + best, key=key)
            task_samples = sorted_samples[:self.num_offsprings]
            selected += [task_samples]
            next_best_samples += [task_samples[:1]]
            print('Selected sample for task %f' % task)
            for i, s in enumerate(task_samples):
                print("%d (%.6f) : %s from %d" % (i, s.evaluate(task),
                                                  s, s.iteration))

        # Update the model
        self.model.update(selected)
        print('-' * 80)
        print(str(self.model))
        print('-' * 80)
        return next_best_samples

    def __str__(self):
        return "[ParameterizedSolver on %s]" % self.prob
