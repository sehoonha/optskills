import numpy as np
from sample import Sample
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


class Sampler(object):
    def __init__(self, _prob, _ntasks, _mean_type):
        self.name = 'SAMPLER'
        self.prob = _prob
        self.n = _ntasks
        self.tasks = np.linspace(0.0, 1.0, self.n)

        self.eval_counter = 0
        self.observers = []

        self.solve()
        # exit(0)

        for t in self.tasks:
            self.view(t)
        plt.show()
        exit(0)

    def add_observer(self, o):
        self.observers += [o]

    def solve(self):
        dim = self.prob.dim
        num_samples = 5000
        fp = open('sample_data_weier_v2.csv', 'w+')

        for i in range(num_samples):
            pt = (np.random.rand(dim) - 0.5) * 2.0
            s = Sample(pt, self.prob)
            values = [s.evaluate(t) for t in self.tasks]
            print('sample %d at %s' % (i, pt))
            row = ", ".join([str(x) for x in [i] + list(pt) + list(values)])
            fp.write(row + "\n")
        fp.close()

    def view(self, task=None):
        # task = 0.0
        data = []
        with open('sample_data_weier_v2.csv') as fp:
            for line in fp.readlines():
                try:
                    data.append([float(x.strip()) for x in line.split(',')])
                except ValueError, e:
                    print('Exception!! %s' % e)
                    print('Line = %s' % line)

        j = 1
        j += self.prob.dim
        j += int(task / 0.2 + 0.001)
        indices = [i for i in range(len(data)) if data[i][j] < 0.2]
        x = [data[i][1] for i in indices]
        y = [data[i][2] for i in indices]
        if self.prob.dim >= 3:
            z = [data[i][3] for i in indices]
        else:
            z = [0.0 for i in indices]
        v = [data[i][j] for i in indices]
        print('Task: %s' % task)
        print('x: %s' % x[:100])
        print('y: %s' % y[:100])
        print('z: %s' % z[:100])
        print('v: %s' % v[:100])
        # x = [row[1] for row in data]
        # y = [row[2] for row in data]
        # z = [row[3] for row in data]
        # v = [row[j] for row in data]
        # print x, y, z

        # p = np.array([-0.31728872, -0.63723352, 0.54830323])  # Bow
        # q = np.array([0.09712741, 0.0908548, -0.10935184])
        # p = np.array([-0.08040776, 0.17626669, 0.86495485])  # Jump
        # q = np.array([0.38804371, 0.29981183, -0.33696449])
        # p = np.array([-0.5, -0.5, 0.0])  # Mirror
        # q = np.array([0.5, -0.5, 0.0])
        # p = np.array([0.79037601, -0.54017897, 0.0])  # Bent cigar
        # q = np.array([-0.12406584, 0.95925613, 0.0])
        # p = np.array([-0.54817862, 0.12114025, 0.0])  # weierstrass
        # q = np.array([1.0666236, -0.04632657, 0.0])
        p = np.array([-0.5, -0.1, 0.0])  # weierstrass v2
        q = np.array([0.1, 0.2, 0.0])
        q += p

        r = (1 - task) * p + task * q


        matplotlib.interactive(False)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.scatter(x, y, z,
                          cmap=cm.coolwarm,
                          norm=matplotlib.colors.LogNorm(),
                          c=v)
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]])
        ax.scatter([p[0], q[0], r[0]], [p[1], q[1], r[1]],
                   [p[2], q[2], r[2]], c=[0.0, task, 1.0])

        fig.colorbar(surf, shrink=0.5, aspect=5, orientation='horizontal')
        plt.title('Task %.4f' % task)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        # ax.view_init(elev=50.0, azim=60.0)
        # ax.view_init(elev=20.0, azim=140.0)
        ax.view_init(elev=80.0, azim=10.0)
        # plt.show()
        # exit(0)

    def num_evals(self):
        return self.prob.eval_counter

    def values(self):
        return [0.0] * self.n

    def __str__(self):
        return "[DirectSolver on %s]" % self.prob
