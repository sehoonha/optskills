import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, STR
import phase_controller
import gp_walk_poses


class GPWalk(SimProblem):
    def __init__(self):
        super(GPWalk, self).__init__('urdf/BioloidGP/BioloidGP.URDF',
                                     fps=5000.0)
        self.__init__simulation__()

        self.dim = 5
        self.eval_counter = 0  # Well, increasing when simulated
        self.params = None

    def __init__simulation__(self):
        self.init_state = self.skel().x
        self.init_state[0] = -0.50 * 3.14
        self.init_state[4] = 0.230
        self.init_state[5] = 0.230

        # self.init_state = gp_walk_poses.targets[0]

        # self.set_init_state('r_foot', 0.5)
        self.set_init_state('r_arm', 0.3)
        self.set_init_state('r_hand', -0.3)
        self.reset()
        self.controller = phase_controller.PhaseController(self.world)
        self.controller.loop_phase_index = 3
        self.controller.callback = self.balance

        self.set_params(np.array([0.14, 0.3, -0.15, -0.1, -0.2]))
        self.controller.reset()

    def balance(self):
        Cd = self.skel().Cdot
        bal = Cd[2]

        l0 = self.skel().dof_index('l_thigh')
        r0 = self.skel().dof_index('r_thigh')
        l1 = self.skel().dof_index('l_shin')
        r1 = self.skel().dof_index('r_shin')
        l2 = self.skel().dof_index('l_heel')
        r2 = self.skel().dof_index('r_heel')

        qhat = np.array(self.controller.phase().target)
        bal *= 1.0
        # print('t: %0.4f bal: %.6f' % (self.world.t, bal))
        qhat[l0] -= bal * 1.0
        qhat[r0] -= bal * 1.0
        qhat[l1] -= bal * 1.0
        qhat[r1] -= bal * 1.0
        qhat[l2] -= bal * 1.0
        qhat[r2] -= bal * 1.0

        self.controller.pd.target = qhat

    def simulate(self, sample):
        self.eval_counter += 1

        self.reset()
        self.set_params(sample)
        while not self.terminated():
            self.step()
        # print 'result:', self.params, self.collect_result()
        return self.collect_result()

    def evaluate(self, result, task):
        w = task
        # Calculate the validity of P (swing foot location)
        P = result['P']
        lo = np.array([0.02, 0.002, 0.06])
        hi = np.array([0.02, 0.002, 0.16])
        P_hat = lo * (1 - w) + hi * w
        weight = np.array([1.0, 1.0, 5.0]) * 2.0
        obj = norm((P - P_hat) * weight) ** 2

        # Calculate the balance penaly
        Cx = result['C'][0]
        Cdotx = result['Cdot'][0]
        b_penalty = (Cx * 2.0) ** 2 + (Cdotx * 1.0) ** 2

        # Calculate parameter penalty
        params = result['params']
        penalty = 0.0
        if params is not None:
            for i in range(self.dim):
                v = params[i]
                penalty += max(0.0, v - 1.0) ** 2
                penalty += min(0.0, v - (-1.0)) ** 2

        return obj + b_penalty + penalty

    def set_random_params(self):
        pass

    def set_params(self, x):
        self.params = x
        w = (x - (-1.0)) / 2.0  # Change to 0 - 1 Scale
        lo = np.array([-0.2, -3.0, -3.0, -3.0, -3.0])
        hi = np.array([0.2, 3.0, 3.0, 3.0, 3.0])
        params = lo * (1 - w) + hi * w
        (q0, q1, q2, q3, q4) = params
        # print q0, q1, q2, q3, q4

        self.reset()
        self.controller.clear_phases()
        for t, q in zip(gp_walk_poses.durations, gp_walk_poses.targets):
            self.controller.add_phase_from_pose(t, q)

    def collect_result(self):
        res = {}
        res['C'] = self.skel().C
        res['Cdot'] = self.skel().Cdot
        res['P'] = self.skel().body('l_foot').C
        res['params'] = self.params
        return res

    def terminated(self):
        return (self.world.t > 5.0)

    def __str__(self):
        res = self.collect_result()
        status = ""
        status += '[GPWalk at %.4f' % self.world.t
        status += '(%d)' % self.controller.phase_index
        # if self.params is not None:
        #     status += ' params = %s ' % self.params
        for key, value in self.collect_result().iteritems():
            if key in set(['params']):
                continue
            if hasattr(value, '__len__'):
                status += ' %s : %s' % (key, STR(value, 3))
            else:
                status += ' %s : %s' % (key, value)

            # if key == 'C':
            #     status += ' %s : %s' % (key, STR(value, 3))
            # elif key == 'params':
            #     status += ' %s : %s' % (key, STR(value, 4))
            # else:
            #     status += ' %s : %s' % (key, value)
        status += ' value = {'
        tasks = np.linspace(0.0, 1.0, 6)
        values = [self.evaluate(res, t) for t in tasks]
        status += ' '.join(['%.4f' % v for v in values])
        status += '}]'
        return status

    def __repr__(self):
        return 'problems.GPWalk()'
