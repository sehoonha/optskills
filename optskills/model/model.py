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

    def generate(self):
        pass

    def update(self, samples):
        # samples is a two-dimensional array of samples
        # sample[i][j] = a jth good sample for the ith task
        pass

    def update_mean(self, samples):
        pass

    def update_covs(self, samples):
        pass

    def __str__(self):
        cov_strings = ["\n%s" % c for c in self.covs]
        return "[Model %s / %s]" % (self.mean, " ".join(cov_strings))
