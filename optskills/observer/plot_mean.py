import numpy as np
import matplotlib.pyplot as plt


class PlotMean(object):
    def __init__(self, mean_type):
        self.mean_type = mean_type
        self.data = []
        self.iter_samples = []

    def notify_init(self, solver, model):
        pass

    def notify_solve(self, solver, model):
        print('PlotMean.notify_solve')
        legends = []
        n = len(self.data)
        colors = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
        fig = plt.figure()
        fig.set_size_inches(12, 12)

        for i, pts in enumerate(self.data):
            legends += ['Mean %02d' % i]
            p0, p1 = pts
            color = next(colors)
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color)
            if i < len(self.iter_samples):
                samples = self.iter_samples[i]
                x = [s[0] for s in samples]
                y = [s[1] for s in samples]
                plt.plot(x, y, 'o', color=color)
                legends += ['Samples %02d' % i]

        plt.legend(legends)
        plt.axes().set_aspect('equal', 'datalim')
        plt.axes().set_xlim(-1.0, 1.0)
        plt.axes().set_ylim(-1.0, 1.0)
        # plt.show()
        plt.savefig('plot_mean.png')

    def notify_step(self, solver, model):
        if self.mean_type == 'linear':
            self.notify_step_linear(model)
        if hasattr(solver, 'iter_samples'):
            self.iter_samples.append(solver.iter_samples)

    def notify_step_linear(self, model):
        mean = model.mean
        p0 = mean.a
        p1 = mean.a + mean.b
        self.data += [(p0, p1)]
