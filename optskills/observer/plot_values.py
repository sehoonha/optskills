import numpy as np
import matplotlib.pyplot as plt


class PlotValues(object):
    def __init__(self):
        self.data = []

    def notify_solve(self, solver, model):
        legends = ['min', 'mean', 'max']
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        plt.plot([min(vs) for vs in self.data])
        plt.plot([np.mean(vs) for vs in self.data])
        plt.plot([max(vs) for vs in self.data])
        plt.legend(legends)
        # plt.show()
        plt.savefig('plot_values.png')

    def notify_step(self, solver, model):
        values = solver.values()
        self.data += [values]
        print('data: %s' % self.data)
