import numpy as np
import matplotlib.pyplot as plt


class PlotMean(object):
    def __init__(self, mean_type):
        self.mean_type = mean_type
        self.data = []

    def notify_solve(self, solver, model):
        legends = []
        n = len(self.data)
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
        fig = plt.figure()
        fig.set_size_inches(12, 12)

        for i, pts in enumerate(self.data):
            legends += ['Iter %02d' % i]
            p0, p1 = pts
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color=next(colors))
        plt.legend(legends)
        plt.axes().set_aspect('equal', 'datalim')
        # plt.show()
        plt.savefig('plot_mean.png')

    def notify_step(self, solver, model):
        if self.mean_type == 'linear':
            self.notify_step_linear(model)

    def notify_step_linear(self, model):
        mean = model.mean
        p0 = mean.a
        p1 = mean.a + mean.b
        self.data += [(p0, p1)]
