import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, SPDController, STR


class SimJump(SimProblem):
    def __init__(self):
        super(SimJump, self).__init__('skel/fullbody1.skel')
        self.__init__simulation__()

        desc = []
        desc.append([('j_thigh_left', 1.0), ('j_thigh_right', 1.0), ])
        desc.append([('j_shin_left', 1.0), ('j_shin_right', 1.0), ])
        desc.append([('j_heel_left', 1.0), ('j_heel_right', 1.0), ])
        self.desc = desc
        self.dim = len(self.desc)
        self.eval_counter = 0  # Well, increasing when simulated
        self.params = None

    def __init__simulation__(self):
        self.init_state = self.skel().x
        self.init_state[1] = -0.50 * 3.14
        self.init_state[4] = 0.92
        self.init_state[5] = 0.0

        self.reset()
        h = self.world.dt
        print('World time step: %.6f' % h)
        self.controller = SPDController(self.skel(), 400.0, 40.0, h)
        self.controller.target = self.skel().q

    def simulate(self, sample):
        self.eval_counter += 1

        self.reset()
        self.set_params(sample)
        while not self.terminated():
            self.step()
        # print 'result:', self.params, self.collect_result()
        return self.collect_result()

    def evaluate(self, result, task):
        # Calculate the validity of COM
        C = result['C']
        lo = np.array([0.0, 0.10, 0.0])
        hi = np.array([0.0, 0.15, 0.0])
        w = task
        C_hat = lo * (1 - w) + hi * w
        weight = np.array([1.0, 1.0, 1.0]) * 2.0
        obj = norm((C - C_hat) * weight) ** 2

        # Calculate parameter penalty
        params = result['params']
        penalty = 0.0
        if params is not None:
            for i in range(self.dim):
                v = params[i]
                penalty += max(0.0, v - 1.0) ** 2
                penalty += min(0.0, v - (-1.0)) ** 2

        return obj + penalty

    def set_random_params(self):
        # self.set_params(0.45 + 0.1 * np.random.rand(self.dim))
        # self.set_params(2.0 * (np.random.rand(self.dim) - 0.5))
        # self.set_params([0.5, -1.0, 0.7])
        self.set_params([0.5, -0.5, 0.1])

    def set_params(self, x):
        self.params = x
        # ndofs = self.skel().ndofs
        # q = np.array(self.init_state[:ndofs])
        # lo = np.array([-2.0] * ndofs)
        # hi = -lo
        # for i, dofs in enumerate(self.desc):
        #     v = (x[i] - (-1.0)) / 2.0  # Change to 0 - 1 scale
        #     for (d, w) in dofs:
        #         index = d if isinstance(d, int) else self.skel().dof_index(d)
        #         vv = v if w > 0.0 else 1.0 - v
        #         q[index] = lo[index] + (hi[index] - lo[index]) * vv
        # self.controller.target = q
        print 'set_params is empty!'

    def collect_result(self):
        res = {}
        res['C'] = self.skel().C
        res['params'] = self.params
        return res

    def terminated(self):
        return (self.world.t > 10.0)

    def __str__(self):
        return 'Good'

    def __repr__(self):
        return 'problems.SimJump()'
