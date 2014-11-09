import numpy as np
import model
import math
from sample import Sample


class ParameterizedSolver(object):
    def __init__(self, _prob, _ntasks, _mean_type):
        self.name = 'Ours'
        self.prob = _prob
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)
        self.model = model.Model(self.prob.dim, self.tasks, _mean_type)
        self.num_parents = 16  # lambda
        self.num_offsprings = 4  # mu
        self.observers = []
        self.no_counter = 0
        print('model: %s' % self.model)
        print('ParameterizedSolver init OK')

    def add_observer(self, o):
        self.observers += [o]

    def solve(self):
        [o.notify_init(self, self.model) for o in self.observers]
        res = {'result': 'NG'}
        MAX_ITER = 10
        best_samples = []
        for i in range(MAX_ITER):
            next_best_samples = self.solve_step(i, best_samples)
            best_samples = next_best_samples
            [o.notify_step(self, self.model) for o in self.observers]

        [o.notify_solve(self, self.model) for o in self.observers]
        return res

    def solve_step(self, iteration, best_samples):
        print('solver iteration: %d' % iteration)
        print('best samples: %s' % best_samples)

        prev_values = self.values()

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

        samples += best_samples
        # Select samples based on the criteria
        selected = []
        next_best_samples = []
        for task in self.tasks:
            key = lambda s: s.evaluate(task)
            sorted_samples = sorted(samples, key=key)
            task_samples = sorted_samples[:self.num_offsprings]
            selected += [task_samples]

            next_best_samples += task_samples
            # print('Selected sample for task %f' % task)
            # for i, s in enumerate(task_samples):
            #     print("%d (%.6f) : %s from %d" % (i, s.evaluate(task),
            #                                       s, s.iteration))

        next_best_samples = list(set(next_best_samples))

        # Update the model
        self.model.update(selected)

        # Step size control
        curr_values = self.values()
        # If offspring is better than parent
        if sum(curr_values) < sum(prev_values):
            self.model.stepsize *= (math.exp(1.0 / 3.0) ** 0.25)
            print('Updated: YES (NO: %d)' % self.no_counter)
        else:
            self.model.stepsize /= (math.exp(1.0 / 3.0) ** 0.25)
            self.no_counter += 1
            print('Updated: NO (%d)' % self.no_counter)

        print('-' * 80)
        print(str(self.model))
        print('average(values): %.8f' % np.mean(self.values()))
        print('-' * 80)
        return next_best_samples

    def num_evals(self):
        return self.prob.eval_counter

    def values(self):
        saved = self.prob.eval_counter
        sample_values = []
        for task in self.tasks:
            pt = self.model.mean.point(task)
            s = Sample(pt, self.prob)
            v = s.evaluate(task)
            sample_values += [v]

        self.prob.eval_counter = saved
        return sample_values

    def __str__(self):
        return "[ParameterizedSolver on %s]" % self.prob
