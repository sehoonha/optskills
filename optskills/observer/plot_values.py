import numpy as np
import scipy.stats.mstats
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

    def best_value(self):
        return min(self.values)

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

    def plot(self, prob_name='', outputfile=''):
        print('\n' * 3)
        print('plot the experiment values')
        names = self.data.keys()
        print('Problem name = %s' % prob_name)
        print('Solver names = %s' % names)
        fp = open('%s_summary.txt' % outputfile, 'w+')
        colors = ['r', 'b', 'g', 'k']
        plt.ioff()
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        num_trials = 0
        index = 0
        pp = []
        legends = []
        names = self.data.keys()
        names.sort()
        names.reverse()
        # for i in range(len(names) - 1):
        #     lhs = names[i]
        #     rhs = names[i + 1]
        #     if 'cubic' in lhs and 'linear' in rhs:
        #         names[i] = rhs
        #         names[i + 1] = lhs
        print names

        # for name, exp_list in self.data.iteritems():
        for name in names:
            exp_list = self.data[name]
            print('=' * 80)
            fp.write('=' * 80 + '\n')
            # exp_list.sort(key=lambda exp: exp.num_evals())
            exp_list.sort(key=lambda exp: exp.best_value())
            print('the initial exp_list: %s' % exp_list)
            fp.write('the initial exp_list: %s\n' % exp_list)
            if len(exp_list) >= 9:
                print('remove outliers')
                fp.write('remove outliers')
                exp_list = exp_list[1:-1]  # Remove outlier exp
            num_trials = len(exp_list)
            med = exp_list[(num_trials - 1) / 2]
            x = med.evals
            y = med.values
            if 'Ours' in name:
                y = list(np.minimum.accumulate(med.values))

            num = 501 if "Ours" in name else 1201
            y_average = np.zeros(num)
            x2 = np.linspace(0, 12000.0, num)
            if "Our" in name:
                x2 = np.linspace(0, 5000.0, num)

            for exp in exp_list:
                x = exp.evals
                y = exp.values
                y2 = np.array([np.interp(t, x, y) for t in x2])
                y_average += y2 / 11.0
            (x, y) = (x2, y_average)

            # while x[-1] > 5000:
            #     x.pop()
            #     y.pop()
            # for i in range(len(x)):
            #     if x[i] > 5000:
            #         x[i] = 5000
            print 'x:', x
            print 'y:', y
            color = 'r' if 'cubic' in name else 'b'
            if '21' in name:
                color = 'r'
            ls = '--' if 'CMA' in name else '-'
            p = plt.plot(x, y, ls=ls, color=color, linewidth=2)
            pp.append(p[0])
            print('')
            print('Exp name: %s' % name)
            fp.write('Exp name: %s\n' % name)
            print('Median index: %d' % ((num_trials - 1) / 2))
            print('exp_list: %s' % exp_list)
            fp.write('exp_list: %s\n' % exp_list)

            final_iters = [e.evals[-1] for e in exp_list]
            # final_values = [min(e.values) for e in exp_list]
            final_values = [e.values[-1] for e in exp_list]
            geom_mean = scipy.stats.mstats.gmean(final_values)
            print('average final iters: %.1f' % np.mean(final_iters))
            print('average final values: %.8f' % np.mean(final_values))
            print('geometric average final values: %.8f' % geom_mean)
            fp.write('average final iters: %.1f\n' % np.mean(final_iters))
            fp.write('average final values: %.8f\n' % np.mean(final_values))
            fp.write('geometric average final values: %.8f\n' % geom_mean)

            # Plot errorbar as well
            # lo = np.percentile(final_values, 10)
            # mi = np.median(final_values)
            # hi = np.percentile(final_values, 90)
            lo = np.min(final_values)
            mi = np.median(final_values)
            hi = np.max(final_values)
            print ('min iters: %f' % lo)
            print ('median: %f' % mi)
            print ('max iters: %f' % hi)
            fp.write('min iters: %f\n' % lo)
            fp.write('median: %f\n' % mi)
            fp.write('max percentile iters: %f\n' % hi)
            # plt.errorbar(x[-1], y[-1], fmt='o', yerr=[[mi - lo], [hi - mi]],
            #              capsize=20, capthick=2.0, color=colors[index])
            # legends += ['%s {%.6f}' % (name, np.mean(final_values))]
            legends += [name]

            # Final, ugly..
            index += 1

        # plt.plot(self.evals, self.values)
        font = {'size': 28}
        # plt.title('Compare %d Trials on %s' % (num_trials, prob_name),
        t = plt.title('',
                      fontdict={'size': 32})
        t.set_y(0.92)
        font = {'size': 28}
        plt.xlabel('# Samples', fontdict=font)
        plt.ylabel('Cost', fontdict=font)
        # plt.legend(pp, self.data.keys(), numpoints=1, fontsize=20)
        # plt.legend(pp, legends, numpoints=1, fontsize=26,
        plt.legend(pp, legends, fontsize=26,
                   # bbox_to_anchor=(0.15, 0.15))
                   # loc='lower left')
                   loc='upper right')
        plt.tick_params(axis='x', labelsize=22)
        plt.tick_params(axis='y', labelsize=22)
        plt.axes().set_yscale('log')
        (lo, hi) = plt.axes().get_ylim()
        # plt.axes().set_ylim(lo - 0.05, hi + 0.05)
        # plt.axes().set_ylim(lo - 0.05, 10)
        # plt.axes().set_ylim(0.0005, 10)
        # plt.axes().set_ylim(0.0001, 10)  # Jumping
        plt.axes().set_ylim(0.0008, 10)  # Kicking
        # plt.axes().set_ylim(0.0005, 10)  # Walking
        plt.axhline(y=0, color='k')
        # plt.show()
        plt.savefig('%s.png' % outputfile, bbox_inches='tight')
        plt.close(fig)
