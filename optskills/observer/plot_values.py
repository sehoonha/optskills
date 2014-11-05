import numpy as np
import matplotlib.pyplot as plt


class PlotValues(object):
    def __init__(self):
        self.evals = []
        self.values = []

    def notify_solve(self, solver, model):
        legends = ['average']
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        plt.plot(self.evals, self.values)
        plt.legend(legends)
        # plt.show()
        plt.savefig('plot_values.png')

    def notify_step(self, solver, model):
        self.evals += [solver.num_evals()]
        self.values += [np.mean(solver.values())]
        # print('data: %s' % self.data)
