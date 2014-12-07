import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, STR
import phase_controller


class GPStep(SimProblem):
    def __init__(self):
        super(GPStep, self).__init__('urdf/BioloidGP/BioloidGP.URDF')
        self.__init__simulation__()

        self.dim = 4
        self.eval_counter = 0  # Well, increasing when simulated
        self.params = None

    def __init__simulation__(self):
        self.init_state = self.skel().x
        self.init_state[0] = -0.50 * 3.14
        self.init_state[4] = 0.230
        self.init_state[5] = 0.230
        self.reset()
        self.controller = phase_controller.PhaseController(self.world)

        # The first phase
        phase = self.controller.add_phase_from_now(0.5)
        phase.set_target('l_thigh', 2.0)

        # The second phase
        phase = self.controller.add_phase_from_prev(0.3)
        phase.set_target('r_thigh', -2.0)
        self.controller.reset()

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
        pass

    def collect_result(self):
        res = {}
        res['C'] = self.skel().C
        res['params'] = self.params
        return res

    def terminated(self):
        return (self.world.t > 1.0)

    def __str__(self):
        res = self.collect_result()
        status = ""
        status += '[GPStep at %.4f' % self.world.t
        status += '(%d)' % self.controller.phase_index
        # if self.params is not None:
        #     status += ' params = %s ' % self.params
        for key, value in self.collect_result().iteritems():
            if key == 'C':
                status += ' %s : %s' % (key, STR(value, 3))
            elif key == 'params':
                status += ' %s : %s' % (key, STR(value, 4))
            else:
                status += ' %s : %s' % (key, value)
        status += ' value = {'
        tasks = np.linspace(0.0, 1.0, 6)
        values = [self.evaluate(res, t) for t in tasks]
        status += ' '.join(['%.4f' % v for v in values])
        status += '}]'
        return status

    def __repr__(self):
        return 'problems.GPStep()'
