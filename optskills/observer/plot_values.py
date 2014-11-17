import numpy as np
import matplotlib.pyplot as plt


class Experiment(object):
    def __init__(self):
        self.evals = []
        self.values = []

    def add_point(self, e, v):
        self.evals += [e]
        self.values += [v]

    def num_evals(self):
        return max(self.evals)

    def num_data(self):
        return max(len(self.evals), len(self.values))

    def __repr__(self):
        return 'Exp(%d, %.6f)' % (self.evals[-1], self.values[-1])


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

    def notify_step(self, solver, model):
        e = solver.num_evals()
        v = np.mean(solver.values())
        self.exp.add_point(e, v)

    # def plot(self):
    #     print('\n' * 3)
    #     print('plot the experiment values')
    #     names = self.data.keys()
    #     print('Solver names = %s' % names)

    #     fig = plt.figure()
    #     fig.set_size_inches(18.5, 10.5)
    #     num_trials = 0
    #     for name, exp_list in self.data.iteritems():
    #         num_trials = len(exp_list)
    #         n = min([e.num_data() for e in exp_list])
    #         x = exp_list[0].evals[:n]
    #         y = []
    #         for i in range(n):
    #             i_values = [e.values[i] for e in exp_list]
    #             avg = np.mean(i_values)
    #             y += [avg]
    #         plt.plot(x, y)

    #         # # Plot errorbar as well
    #         last_values = [e.values[n - 1] for e in exp_list]
    #         print('%s: %.8f' % (name, np.mean(last_values)))
    #         print('last_values: %s' % last_values)
    #         # lo = np.percentile(last_values, 20)
    #         # mi = np.mean(last_values)
    #         # hi = np.percentile(last_values, 80)
    #         # plt.errorbar(x[n - 1], y[n - 1], yerr=[[mi - lo], [hi - mi]])
    #     # plt.plot(self.evals, self.values)
    #     font = {'size': 24}
    #     plt.title('Compare %d Trials' % num_trials, fontdict=font)
    #     font = {'size': 20}
    #     plt.xlabel('The number of sample evaluations', fontdict=font)
    #     plt.ylabel('The average error of mean segments', fontdict=font)
    #     plt.legend(self.data.keys(), fontsize=20)
    #     # plt.show()
    #     plt.savefig('plot_values.png')

    def plot(self, prob_name=''):
        print('\n' * 3)
        print('plot the experiment values')
        names = self.data.keys()
        print('Problem name = %s' % prob_name)
        print('Solver names = %s' % names)
        colors = ['r', 'g', 'b']
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        num_trials = 0
        index = 0
        pp = []
        for name, exp_list in self.data.iteritems():
            exp_list.sort(key=lambda exp: exp.num_evals())
            num_trials = len(exp_list)
            med = exp_list[(num_trials - 1) / 2]
            x = med.evals
            y = med.values
            p = plt.plot(x, y, color=colors[index])
            pp.append(p[0])
            print('')
            print('Exp name: %s' % name)
            print('Median index: %d' % ((num_trials - 1) / 2))
            print('exp_list: %s' % exp_list)

            final_iters = [e.evals[-1] for e in exp_list]
            final_values = [e.values[-1] for e in exp_list]
            print('average final iters: %.1f' % np.mean(final_iters))
            print('average final values: %.8f' % np.mean(final_values))

            # Plot errorbar as well
            lo = np.percentile(final_iters, 10)
            mi = x[-1]
            hi = np.percentile(final_iters, 90)
            print ('10%% percentile iters: %d' % lo)
            print ('median: %d' % mi)
            print ('90%% percentile iters: %d' % hi)
            plt.errorbar(x[-1], y[-1], fmt='o', xerr=[[mi - lo], [hi - mi]],
                         capsize=20, capthick=2.0, color=colors[index])

            # Final, ugly..
            index += 1

        # plt.plot(self.evals, self.values)
        font = {'size': 24}
        plt.title('Compare %d Trials on %s' % (num_trials, prob_name),
                  fontdict=font)
        font = {'size': 20}
        plt.xlabel('The number of sample evaluations', fontdict=font)
        plt.ylabel('The error of mean segments', fontdict=font)
        plt.legend(pp, self.data.keys(), numpoints=1, fontsize=20)
        plt.axes().set_ylim(-0.1, 1.0)
        plt.axhline(y=0, color='k')
        # plt.show()
        plt.savefig('plot_values.png')
