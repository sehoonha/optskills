import numpy as np
from numpy.linalg import norm
import model
import math
from sample import Sample
import copy


class ParameterizedSolver(object):
    def __init__(self, _prob, _ntasks, _mean_type, _alg=''):
        self.name = 'Ours(%s)' % _alg
        self.alg = _alg
        self.prob = _prob
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)
        self.model = model.Model(self.prob.dim, self.tasks, _mean_type)
        self.num_parents = 16
        # self.num_parents = 32
        # lambda
        self.num_offsprings = 8  # mu
        self.observers = []
        self.no_counter = 0
        print('model: %s' % self.model)
        print('algorithm: [%s]' % self.alg)
        print('ParameterizedSolver init OK')

    def add_observer(self, o):
        self.observers += [o]

    def solve(self):
        [o.notify_init(self, self.model) for o in self.observers]
        res = {'result': 'NG'}
        # MAX_ITER = int(5000 / (self.num_parents + self.n)) + 50
        MAX_ITER = 1000
        print('MAX_ITER: %d', MAX_ITER)
        self.mean_values, self.mean_samples = self.evaluate_model(self.model,
                                                                  -1)
        [o.notify_step(self, self.model) for o in self.observers]
        best_samples = self.mean_samples
        for i in range(MAX_ITER):
            next_best_samples = self.solve_step(i, best_samples)
            best_samples = next_best_samples
            [o.notify_step(self, self.model) for o in self.observers]
            # if np.mean(self.mean_values) < 0.001:
            #     break
            if self.prob.eval_counter > 5000:
                break
            # if self.model.stepsize < 0.00001:
            #     break
        [o.notify_solve(self, self.model) for o in self.observers]
        return res

    def solve_step(self, iteration, best_samples):
        print('\n' * 2)
        print('solver iteration: %d' % iteration)
        print('previous mean values: %s' % self.mean_values)
        print('previous mean samples: %s' % self.mean_samples)
        # print('best samples: %s' % best_samples)

        # Generate the population based on the current model
        samples = self.generate_samples(iteration)
        self.iter_samples = samples

        # Add previous best samples, for stability
        samples += best_samples

        # Select samples based on the criteria
        selected = self.select_samples(samples)

        # Save a compact set of selected samples
        next_best_samples = sum(selected, [])  # Flatten the selected list
        next_best_samples = list(set(next_best_samples))

        # Update the model
        curr_model = copy.deepcopy(self.model)
        curr_model.update(selected, self.alg)
        curr_param = curr_model.mean.params()
        self_param = self.model.mean.params()
        print('curr_param = %s' % curr_param)
        print('self_param = %s' % self_param)
        print('dist = %.8f' % norm(curr_param - self_param))
        if norm(curr_param - self_param) < 1e-10:
            print('Skip evaluation: same')
            curr_mean_values = copy.deepcopy(self.mean_values)
            curr_mean_samples = copy.deepcopy(self.mean_samples)
        else:
            curr_mean_values, curr_mean_samples = self.evaluate_model(
                curr_model, iteration)

        is_better = np.mean(curr_mean_values) < np.mean(self.mean_values)
        print('self.mean_values: %.8f' % np.mean(self.mean_values))
        print('curr_mean_values: %.8f' % np.mean(curr_mean_values))

        stepsize = self.model.stepsize
        p_succ = self.model.p_succ

        if 'step_success' in self.alg:
            (stepsize, p_succ) = self.model.update_stepsize_success(stepsize,
                                                                    is_better,
                                                                    p_succ)
        else:
            (stepsize, p_succ) = self.model.update_stepsize_1_5th(stepsize,
                                                                  is_better,
                                                                  p_succ)
        # If offspring is better than parent
        if is_better:
            print('Updated: YES (NO: %d)' % self.no_counter)
        else:
            self.no_counter += 1
            print('Updated: NO (NO: %d)' % self.no_counter)

        # Finalize
        if is_better:
            self.model = curr_model
            self.mean_values = curr_mean_values
            self.mean_samples = curr_mean_samples
            self.model.volumns = self.mean_values
        self.model.stepsize = stepsize  # Always update stepsize
        self.model.p_succ = p_succ

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
            params = self.model.generate_params(self.alg)
            s = Sample(params, self.prob)
            s.iteration = iteration
            self.prob.reset()
            s.simulate()
            samples += [s]
            # Debuging
            j = self.model.debug_last_generate_index
            print("%s (from %d) %s" % (i, j, s))
            print('  >> params: %s' % s)
            print('  >> result: %s' % s.result)
            print('  >> values: %s' % ([s.evaluate(t) for t in self.tasks]))
            s2 = Sample(params, self.prob)
            s2.simulate()
            print('  >> params: %s' % s2)
            print('  >> result: %s' % s2.result)
            print('  >> values: %s' % ([s2.evaluate(t) for t in self.tasks]))
            s3 = Sample(params, self.prob)
            s3.simulate()
            print('  >> params: %s' % s3)
            print('  >> result: %s' % s3.result)
            print('  >> values: %s' % ([s3.evaluate(t) for t in self.tasks]))
            print('\n\n')
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
                break
        return selected

    def evaluate_model(self, model, iteration):
        mean_samples = []
        mean_values = []

        for i, task in enumerate(self.tasks):
            pt = model.mean.point(task)
            s = Sample(pt, self.prob)
            s.iteration = iteration

            # #############
            # Hacky init T.T
            if i == 0:
                saved = self.prob.eval_counter
                # print 'null evaluate'
                s2 = Sample(pt, self.prob)
                s2.evaluate(task)
                self.prob.eval_counter = saved
            # ###############
            v = s.evaluate(task)
            print 'evaluate::', i, task, pt, v
            print('  >> result: %s' % s.result)
            print('\n\n')
            mean_samples += [s]
            mean_values += [v]
        return mean_values, mean_samples

    def evaluate_model_approx(self, model, iteration, selected):
        mean_samples = []
        mean_values = []
        for i, task in enumerate(self.tasks):
            s = selected[i][0]
            v = s.evaluate(task)
            mean_samples += [s]
            mean_values += [v]
        return mean_values, mean_samples

    def num_evals(self):
        return self.prob.eval_counter

    def values(self):
        return self.mean_values
        # saved = self.prob.eval_counter
        # sample_values = []
        # for task in self.tasks:
        #     pt = self.model.mean.point(task)
        #     s = Sample(pt, self.prob)
        #     v = s.evaluate(task)
        #     sample_values += [v]

        # self.prob.eval_counter = saved
        # return sample_values

    def __str__(self):
        return "[ParameterizedSolver on %s]" % self.prob
