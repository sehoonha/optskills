import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, PDController


class GPBow(SimProblem):
    def __init__(self):
        super(GPBow, self).__init__('urdf/BioloidGP/BioloidGP.URDF')
        self.__init__simulation__()

        desc = []
        desc.append([('l_thigh', 1.0), ('r_thigh', 1.0), ])
        desc.append([('l_shin', 1.0), ('r_shin', 1.0), ])
        desc.append([('l_heel', 1.0), ('r_heel', 1.0), ])
        self.desc = desc
        self.dim = len(self.desc)
        self.eval_counter = 0  # Well, increasing when simulated
        self.params = None

    def __init__simulation__(self):
        self.init_state = self.skel().x
        self.init_state[0] = -0.50 * 3.14
        self.init_state[4] = 0.230
        self.init_state[5] = 0.230
        self.reset()
        self.controller = PDController(self.skel(), 60, 1.0, 0.5)
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
        C = result['C']
        lo = np.array([0.0, 0.10, 0.0])
        hi = np.array([0.0, 0.15, 0.0])
        w = task
        C_hat = lo * (1 - w) + hi * w
        weight = np.array([10.0, 1.0, 10.0])
        return norm((C - C_hat) * weight) ** 2

    def set_random_params(self):
        self.set_params(0.45 + 0.1 * np.random.rand(self.dim))

    def set_params(self, x):
        self.params = x
        q = np.array(self.init_state[:self.skel().ndofs])
        lo = self.skel().q_lo
        hi = self.skel().q_hi
        for i, dofs in enumerate(self.desc):
            v = x[i]
            for (d, w) in dofs:
                index = d if isinstance(d, int) else self.skel().dof_index(d)
                vv = v if w > 0.0 else 1.0 - v
                q[index] = lo[index] + (hi[index] - lo[index]) * vv
        self.controller.target = q

    def collect_result(self):
        ret = {}
        ret['C'] = self.skel().C
        return ret

    def terminated(self):
        return (self.world.t > 0.5)

    def __str__(self):
        res = self.collect_result()
        status = ""
        status += '[GPBow at %.4f' % self.world.t
        if self.params is not None:
            status += ' params = %s ' % self.params
        for key, value in self.collect_result().iteritems():
            status += ' %s : %s' % (key, value)
        status += ' value = {'
        tasks = np.linspace(0.0, 1.0, 6)
        values = [self.evaluate(res, t) for t in tasks]
        status += ' '.join(['%.6f' % v for v in values])
        status += '}]'
        return status
