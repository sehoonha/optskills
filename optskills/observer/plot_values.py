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
    def __init__(self, _datafile=None):
        self.data = {}

        self.datafile = _datafile
        if self.datafile is not None:
            self.fout = open(self.datafile, 'w+')
        else:
            self.fout = None

    def notify_init(self, solver, model):
        self.exp = Experiment()
        # Log to file
        if self.fout is not None:
            self.fout.write('Experiment, %s\n' % solver.name)
            self.fout.flush()

    def notify_solve(self, solver, model):
        self.add_exp(solver.name)
        # if name not in self.data:
        #     self.data[name] = []

        # self.data[name] += [self.exp]
        # self.exp = None

        # Log to file
        if self.fout is not None:
            self.fout.write('Solved\n')
            self.fout.flush()

    def add_exp(self, name):
        if name not in self.data:
            self.data[name] = []

        self.data[name] += [self.exp]
        self.exp = None

    def notify_step(self, solver, model):
        e = solver.num_evals()
        v = np.mean(solver.values())
        self.exp.add_point(e, v)
        # Log to file
        if self.fout is not None:
            self.fout.write('%d, %f\n' % (e, v))
            self.fout.flush()

    def load(self, filename):
        self.exp = None
        name = None
        with open(filename) as fin:
            for line in fin.readlines():
                tokens = line.split(', ')
                if 'Experiment' in line:
                    if self.exp is not None:
                        self.add_exp(name)
                    self.exp = Experiment()
                    name = tokens[1]
                elif 'Solved' in line:
                    self.add_exp(name)
                else:
                    e = int(tokens[0])
                    v = float(tokens[1])
                    self.exp.add_point(e, v)
        if self.exp is not None:
            self.add_exp(name)

    def save(self, filename):
        with open(filename, "w+") as fout:
            for name, exp_list in self.data.iteritems():
                for exp in exp_list:
                    fout.write('Experiment, %s\n' % name)
                    for e, v in zip(exp.evals, exp.values):
                        fout.write('%d, %f\n' % (e, v))
                    fout.write('Solved\n')
                    fout.flush()

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
            print 'x:', x
            print 'y:', y
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
        (lo, hi) = plt.axes().get_ylim()
        plt.axes().set_ylim(lo - 0.05, hi + 0.05)
        plt.axhline(y=0, color='k')
        # plt.show()
        plt.savefig('plot_values.png')
