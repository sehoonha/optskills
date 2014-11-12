import numpy as np
import model
import math
from sample import Sample
import copy


class ParameterizedSolver(object):
    def __init__(self, _prob, _ntasks, _mean_type):
        self.name = 'Ours'
        self.prob = _prob
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)
        self.model = model.Model(self.prob.dim, self.tasks, _mean_type)
        self.num_parents = 16
        # lambda
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
        self.mean_values, self.mean_samples = self.evaluate_model(self.model,
                                                                  -1)
        best_samples = self.mean_samples
        for i in range(MAX_ITER):
            next_best_samples = self.solve_step(i, best_samples)
            best_samples = next_best_samples
            [o.notify_step(self, self.model) for o in self.observers]
            if np.mean(self.mean_values) < 0.0001:
                break
        [o.notify_solve(self, self.model) for o in self.observers]
        return res

    def solve_step(self, iteration, best_samples):
        print('\n' * 2)
        print('solver iteration: %d' % iteration)
        print('previous mean values: %s' % self.mean_values)
        print('previous mean samples: %s' % self.mean_samples)
        print('best samples: %s' % best_samples)

        # Generate the population based on the current model
        samples = self.generate_samples(iteration)

        # Add previous best samples, for stability
        samples += best_samples

        # Select samples based on the criteria
        selected = self.select_samples(samples)

        # Save a compact set of selected samples
        next_best_samples = sum(selected, [])  # Flatten the selected list
        next_best_samples = list(set(next_best_samples))

        # Update the model
        curr_model = copy.deepcopy(self.model)
        curr_model.update(selected)
        curr_mean_values, curr_mean_samples = self.evaluate_model(curr_model,
                                                                  iteration)

        is_better = np.mean(curr_mean_values) < np.mean(self.mean_values)
        print('self.mean_values: %.8f' % np.mean(self.mean_values))
        print('curr_mean_values: %.8f' % np.mean(curr_mean_values))

        stepsize = self.model.stepsize
        # If offspring is better than parent
        if is_better:
            # self.model.stepsize *= (math.exp(1.0 / 3.0) ** 0.25)
            stepsize *= (math.exp(1.0 / 3.0))
            print('Updated: YES (NO: %d)' % self.no_counter)
        else:
            stepsize /= (math.exp(1.0 / 3.0) ** 0.25)
            self.no_counter += 1
            print('Updated: NO (NO: %d)' % self.no_counter)

        # Finalize
        if is_better:
            self.model = curr_model
            self.mean_values = curr_mean_values
            self.mean_samples = curr_mean_samples
        self.model.stepsize = stepsize  # Always update stepsize

        # Print out some information
        print('-' * 80)
        print(str(self.model))
        print('eval_counter: %d' % self.prob.eval_counter)
        print('average(values): %.8f' % np.mean(self.mean_values))
        print('-' * 80)
        return next_best_samples

    def generate_samples(self, iteration):
        samples = []
        # Generate all samples
        for i in range(self.num_parents):
            # Generate params from the model and make a sample
            params = self.model.generate_params()
            s = Sample(params, self.prob)
            s.iteration = iteration
            s.simulate()
            samples += [s]
            # Debuging
            j = self.model.debug_last_generate_index
            print("%s (from %d) %s" % (i, j, s))
        return samples

    def select_samples(self, samples):
        # Select samples based on the criteria
        selected = []
        for task in self.tasks:
            key = lambda s: s.evaluate(task)
            sorted_samples = sorted(samples, key=key)
            task_samples = sorted_samples[:self.num_offsprings]
            selected += [task_samples]

            print('Selected sample for task %f' % task)
            for i, s in enumerate(task_samples):
                print("%d (%.6f) : %s from %d" % (i, s.evaluate(task),
                                                  s, s.iteration))
        return selected

    def evaluate_model(self, model, iteration):
        mean_samples = []
        mean_values = []
        for task in self.tasks:
            pt = model.mean.point(task)
            s = Sample(pt, self.prob)
            s.iteration = iteration
            v = s.evaluate(task)
            mean_samples += [s]
            mean_values += [v]
        return mean_values, mean_samples

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
