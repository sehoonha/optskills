import numpy as np
from mean import Linear
from cov import Cov


class Model(object):
    def __init__(self, dim, tasks, mean_type='linear'):
        self.dim = dim
        self.tasks = tasks
        self.n = len(tasks)

        # Create a mean function
        if mean_type == 'linear':
            self.mean = Linear(dim, tasks)

        # Populate a set of covarience matrices
        self.covs = []
        for task in self.tasks:
            center = self.mean.point(task)
            self.covs += [Cov(dim, center)]

        # Populate a set of volumns
        self.volumns = [1.0] * self.n

    def generate_params(self):
        prob = np.array(self.volumns) / sum(self.volumns)
        # i = picked covariance matrix
        i = np.random.choice(range(self.n), p=prob)
        self.debug_last_generate_index = i
        return self.covs[i].generate_params()

    def update(self, samples):
        # samples is a two-dimensional array of samples
        # sample[i][j] = a jth good sample for the ith task
        self.update_mean(samples)
        self.update_covs(samples)
        self.volumns = []
        for task, samples_for_task in zip(self.tasks, samples):
            s = samples_for_task[0]
            v = s.evaluate(task)
            self.volumns += [v]

    def update_mean(self, samples):
        pts = []
        for selected_for_task in samples:
            # m = np.mean(selected_for_task, axis=0)
            m = selected_for_task[0]
            pts += [m]
        self.mean.fit(pts)

    def update_covs(self, samples):
        mean_pts = [self.mean.point(t) for t in self.tasks]
        self.covs = []
        for m, selected_for_task in zip(mean_pts, samples):
            pts = np.matrix(selected_for_task)
            self.covs += [Cov(self.dim, m, pts)]

    def __str__(self):
        cov_strings = ["\n%s" % c for c in self.covs]
        return "[Model %s %s / %s]" % (self.mean, self.volumns,
                                       " ".join(cov_strings))