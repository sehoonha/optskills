import math
import numpy as np
from mean import Linear, Cubic
from cov import Cov


class Model(object):
    def __init__(self, dim, tasks, mean_type='linear'):
        self.dim = dim
        self.tasks = tasks
        self.n = len(tasks)
        self.mean_type = mean_type

        print('mean_type: %s' % mean_type)
        # exit(0)

        # Create a mean function
        if mean_type == 'linear':
            self.mean = Linear(dim, tasks)
        elif mean_type == 'cubic':
            self.mean = Cubic(dim, tasks)

        # Populate a set of covarience matrices
        self.covs = []
        self.paths = []
        for task in self.tasks:
            self.paths += [np.zeros(self.dim)]
            center = self.mean.point(task)
            self.covs += [Cov(dim, center)]

        # Stepsize, externally controlled
        self.stepsize = 1.0
        self.p_succ_target = 2.0 / 11.0
        self.p_succ = self.p_succ_target

        # Populate a set of volumns
        self.volumns = [1.0] * self.n
        print 'n:', self.n
        print 'volumns:', self.volumns

    def generate_params(self, alg=''):
        if 'draw_uniform' in alg:
            prob = np.array([float(1.0 / self.n)] * self.n)
        else:
            prob = np.array(self.volumns) / sum(self.volumns)

        # print 'probs:', prob
        # i = picked covariance matrix
        i = np.random.choice(range(self.n), p=prob)
        self.debug_last_generate_index = i
        return self.covs[i].generate_params(self.stepsize)

    def mean_centers(self):
        return [self.mean.point(t) for t in self.tasks]

    def update(self, samples, alg=''):
        # samples is a two-dimensional array of samples
        # sample[i][j] = a jth good sample for the ith task

        prev_centers = self.mean_centers()

        # Various algorithms of update mean
        if 'mean_best' in alg:
            self.update_mean(samples)
        elif 'mean_avg' in alg:
            self.update_mean_groupmean(samples)
        elif 'mean_all' in alg:
            self.update_mean_all(samples)
        else:
            self.update_mean_randomized(samples)

        self.update_paths(prev_centers)

        if 'cov_rank_1' in alg:
            self.update_covs_rank_1()
        else:
            self.update_covs(samples)

        self.volumns = []
        for task, samples_for_task in zip(self.tasks, samples):
            s = samples_for_task[0]
            v = s.evaluate(task)
            self.volumns += [v]

    def update_mean(self, samples):
        print('update_mean <best>')
        pts = []
        for selected_for_task in samples:
            # m = np.mean(selected_for_task, axis=0)
            m = selected_for_task[0]
            pts += [m]
        self.mean.fit(pts)

    def update_mean_groupmean(self, samples):
        print('update_mean <avg>')
        pts = []
        for selected_for_task in samples:
            m = np.mean(selected_for_task, axis=0)
            pts += [m]
        self.mean.fit(pts)

    def update_mean_all(self, samples):
        print('update_mean <all>')
        pts = []
        xdata = []
        for task, selected_for_task in zip(self.tasks, samples):
            for pt in selected_for_task:
                xdata += [task]
                pts += [pt]
        self.mean.fit(pts, xdata)

    def update_mean_randomized(self, samples):
        print('update_mean <randomized>')
        NUM_TRIALS = 64
        best_estimation, best_params = None, None

        for loop in range(NUM_TRIALS):
            pts = []
            values = []
            for task, selected_for_task in zip(self.tasks, samples):
                nbests = len(selected_for_task)
                i = np.random.choice(range(nbests))
                if loop == 0:
                    i = 0
                s = selected_for_task[i]
                pts += [s]
                values += [s.evaluate(task)]
            self.mean.fit(pts)
            estimated_cost = sum(values) + 10.0 * self.mean.fit_error
            # estimated_cost = sum(values) + 1.0 * self.mean.fit_error
            # print loop, self.mean.params(), ':',
            # print estimated_cost, sum(values), estimated_cost - sum(values)
            if best_estimation is None or estimated_cost < best_estimation:
                best_estimation = estimated_cost
                best_params = self.mean.params()
        self.mean.set_params(best_params)

    def update_paths(self, prev_centers):
        print('update_path')
        curr_centers = self.mean_centers()
        n = self.dim
        c_c = 2.0 / (n + 2.0)
        p_thresh = 0.44
        for i in range(len(self.tasks)):
            p = self.paths[i]
            y = curr_centers[i] - prev_centers[i]
            if self.p_succ < p_thresh:
                p = (1 - c_c) * p + math.sqrt(c_c * (2.0 - c_c)) * y
            else:
                p = (1 - c_c) * p
            self.paths[i] = p

    def update_covs(self, samples):
        print('update_covs <all>')
        mean_pts = [self.mean.point(t) for t in self.tasks]
        self.covs = []
        for m, selected_for_task in zip(mean_pts, samples):
            pts = np.matrix(selected_for_task)
            self.covs += [Cov(self.dim, m, pts)]

    def update_covs_rank_1(self):
        print('update_covs <rank1>')
        n = self.dim
        c_c = 2.0 / (n + 2.0)
        c_cov = 2.0 / ((n  ** 2) + 2.0)
        p_thresh = 0.44
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            m = self.mean.point(task)
            p = self.paths[i]
            C = self.covs[i].C
            if self.p_succ < p_thresh:
                C = (1 - c_cov) * C + c_cov * np.outer(p, p)
            else:
                C = (1 - c_cov) * C \
                    + c_cov * (np.outer(p, p) + c_c * (2 - c_c) * C)
            self.covs[i] = Cov(self.dim, m, _C=C)

    def update_stepsize_1_5th(self, stepsize, is_better, p_succ):
        print('update_stepsize <1/5th>')
        c_p = 1.0 / 12.0
        lambda_succ = 1.0 if is_better else 0.0
        p_succ = (1 - c_p) * p_succ + c_p * lambda_succ

        if is_better:
            # self.model.stepsize *= (math.exp(1.0 / 3.0) ** 0.25)
            # stepsize *= (math.exp(1.0 / 3.0) ** 0.25)
            stepsize *= (math.exp(1.0 / 3.0))
        else:
            stepsize /= (math.exp(1.0 / 3.0) ** 0.25)
        return (stepsize, p_succ)

    def update_stepsize_success(self, stepsize, is_better, p_succ):
        print('update_stepsize <success>')
        c_p = 1.0 / 12.0
        lambda_succ = 1.0 if is_better else 0.0
        p_succ = (1 - c_p) * p_succ + c_p * lambda_succ
        p_succ_target = self.p_succ_target
        d = 1.0 + self.dim / 2.0
        stepsize = stepsize * math.exp((1 / d) * (
            p_succ - (p_succ_target / (1 - p_succ_target)) * (1 - p_succ)))
        return (stepsize, p_succ)

    def set_stepsize(self, stepsize):
        self.stepsize = stepsize

    def __str__(self):
        cov_strings = ["\n%s" % c for c in self.covs]
        ret = "[Model %s step (%.4f %.4f) %s / %s" % (self.mean,
                                                      self.stepsize,
                                                      self.p_succ,
                                                      self.volumns,
                                                      " ".join(cov_strings))
        ret += "\n".join(["path %s" % p for p in self.paths])
        ret += "]"
        return ret
