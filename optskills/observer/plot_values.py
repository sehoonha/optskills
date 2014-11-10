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
        num_trials = 0
        for name, exp_list in self.data.iteritems():
            num_trials = len(exp_list)
            n = min([e.num_data() for e in exp_list])
            x = exp_list[0].evals[:n]
            y = []
            for i in range(n):
                i_values = [e.values[i] for e in exp_list]
                avg = np.mean(i_values)
                y += [avg]
            plt.plot(x, y)

            # # Plot errorbar as well
            last_values = [e.values[n - 1] for e in exp_list]
            print('%s: %.8f' % (name, np.mean(last_values)))
            print('last_values: %s' % last_values)
            # lo = np.percentile(last_values, 20)
            # mi = np.mean(last_values)
            # hi = np.percentile(last_values, 80)
            # plt.errorbar(x[n - 1], y[n - 1], yerr=[[mi - lo], [hi - mi]])
        # plt.plot(self.evals, self.values)
        font = {'size': 24}
        plt.title('Compare %d Trials' % num_trials, fontdict=font)
        font = {'size': 20}
        plt.xlabel('The number of sample evaluations', fontdict=font)
        plt.ylabel('The average error of mean segments', fontdict=font)
        plt.legend(self.data.keys(), fontsize=20)
        # plt.show()
        plt.savefig('plot_values.png')
