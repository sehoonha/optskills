import numpy as np
import matplotlib.pyplot as plt


class Experiment(object):
    def __init__(self):
        self.evals = []
        self.values = []

    def add_point(self, e, v):
        self.evals += [e]
        self.values += [v]

    def num_data(self):
        return max(len(self.evals), len(self.values))


class PlotValues(object):
    def __init__(self):
        self.data = {}

    def notify_init(self, solver, model):
        self.exp = Experiment()

    def notify_solve(self, solver, model):
        name = solver.name
        if name not in self.data:
            self.data[name] = []

        self.data[name] += [self.exp]
        self.exp = None
        # legends = ['average']
        # fig = plt.figure()
        # fig.set_size_inches(18.5, 10.5)
        # plt.plot(self.evals, self.values)
        # plt.legend(legends)
        # # plt.show()
        # plt.savefig('plot_values.png')

    def notify_step(self, solver, model):
        e = solver.num_evals()
        v = np.mean(solver.values())
        self.exp.add_point(e, v)

        # self.evals += [solver.num_evals()]
        # self.values += [np.mean(solver.values())]
        # # print('data: %s' % self.data)

    def plot(self):
        print('\n' * 3)
        print('plot the experiment values')
        names = self.data.keys()
        print('Solver names = %s' % names)

        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        for name, exp_list in self.data.iteritems():
            n = min([e.num_data() for e in exp_list])
            x = exp_list[0].evals[:n]
            y = []
            for i in range(n):
                avg = np.mean([e.values[i] for e in exp_list])
                y += [avg]
            plt.plot(x, y)
        # plt.plot(self.evals, self.values)
        plt.legend(self.data.keys())
        # plt.show()
        plt.savefig('plot_values.png')
