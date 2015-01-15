import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, STR
import phase_controller
import math


class GPJump(SimProblem):
    def __init__(self):
        super(GPJump, self).__init__('urdf/BioloidGP/BioloidGP.URDF',
                                     fps=1000.0)
        self.__init__simulation__()

        self.dim = 6
        self.eval_counter = 0  # Well, increasing when simulated
        self.params = None

    def __init__simulation__(self):
        self.init_state = self.skel().x
        self.init_state[0] = -0.50 * 3.14
        self.init_state[4] = 0.230
        self.init_state[5] = 0.230
        self.reset()
        self.controller = phase_controller.PhaseController(self.world)
        self.controller.callback = self.balance

        # self.set_params(np.array([0.14, 0.3, -0.15, -0.1, -0.2, 0.2, 0, 0]))
        self.set_params(np.array([0.14, 0.3, -0.15, -0.1, -0.2, 0.2]))
        self.controller.reset()

    def balance(self):
        Cd = self.skel().Cdot
        bal = Cd[2]

        l0 = self.skel().dof_index('l_thigh')
        r0 = self.skel().dof_index('r_thigh')
        l2 = self.skel().dof_index('l_heel')
        r2 = self.skel().dof_index('r_heel')

        qhat = np.array(self.controller.phase().target)
        # bal *= 1.0
        bal *= 0.7
        # print('t: %0.4f bal: %.6f' % (self.world.t, bal))
        qhat[l0] -= bal * 1.0
        qhat[r0] -= bal * 1.0
        qhat[l2] -= bal * 1.0
        qhat[r2] -= bal * 1.0

        self.controller.pd.target = qhat

    def simulate(self, sample):
        self.eval_counter += 1

        self.set_params(sample)
        self.reset()
        while not self.terminated():
            self.step()
        return self.collect_result()

    def evaluate(self, result, task):
        w = task
        # Calculate the validity of C
        C = np.array(result['C'])
        C[1] = result['maxCy']

        lo = np.array([0.0, 0.22, 0.00])
        hi = np.array([0.0, 0.27, 0.00])
        C_hat = lo * (1 - w) + hi * w
        weight = np.array([0.1, 10.0, 0.0])
        obj = norm((C - C_hat) * weight) ** 2

        if result['maxCy'] < 0.2:
            obj += 10.0 * (0.2 - result['maxCy'])

        b_penalty = 0.0
        if result['fallen']:
            t = result['t']
            b_penalty += 0.5 * (2.0 - t)

        # Calculate parameter penalty
        params = result['params']
        penalty = 0.0
        if params is not None:
            for i in range(self.dim):
                v = params[i]
                penalty += max(0.0, v - 1.0) ** 2
                penalty += min(0.0, v - (-1.0)) ** 2

        return obj + b_penalty + penalty
        # return b_penalty

    def set_random_params(self):
        pass

    def set_params(self, x):
        self.params = x
        w = (x - (-1.0)) / 2.0  # Change to 0 - 1 Scale
        lo = np.array([-0.5, -1.5, -0.5, -1.0, -200, -200])
        hi = np.array([1.5, 0.0, 1.5, 1.0, 0, 0])
        params = lo * (1 - w) + hi * w
        (q0, q1, q2, q3, f0, f1) = params
        (q4, q5) = (-0.7, 0.0)
        # print 'q:', q0, q1, q2, q3, f0, f1
        # (q0, q1, q2, q3, f0, f1) = (0.65, -1.3, 0.55, 0.2, -150, -70)
        # (q0, q1, q2, q3, f0, f1) = (0.65, -1.3, 0.55, 0.2, -100, -40)
        # print 'q:', q0, q1, q2, q3, f0, f1
        # (q0, q1, q2, q3, q4) = (0.16, -0.8, -1.0, 0.60, -0.5)

        self.reset()
        self.controller.clear_phases()
        # The first phase - balance
        phase = self.controller.add_phase_from_now(0.7)
        phase.set_target('l_thigh', q0)  # 0.65
        phase.set_target('r_thigh', q0)  # 0.65
        phase.set_target('l_shin', q1)  # -1.3
        phase.set_target('r_shin', q1)  # -1.3
        phase.set_target('l_heel', q2)  # 0.6
        phase.set_target('r_heel', q2)  # 0.6
        phase.set_target('l_shoulder', q4)  # -0.7
        phase.set_target('r_shoulder', q4)  # -0.7

        # The second phase - swing back
        phase = self.controller.add_phase_from_now('no_contact')
        phase.add_virtual_force(['l_foot', 'r_foot'],
                                np.array([0, f0, f1]))   # 0, -150, -55

        # The third phase - swing forward
        phase = self.controller.add_phase_from_now(0.8)
        phase.set_target('l_shoulder', 0.3)  # 0.3
        phase.set_target('r_shoulder', 0.3)  # 0.3
        phase.set_target('l_thigh', q3)  # 0.2
        phase.set_target('r_thigh', q3)  # 0.2
        phase.add_target_offset('l_shin', q5)  # 0.2
        phase.add_target_offset('r_shin', q5)  # 0.2

        # # For the final production
        # phase.terminae = 0.5
        # phase = self.controller.add_phase_from_now(0.8)
        # phase.add_target_offset('l_heel', -0.1)  # 0.2
        # phase.add_target_offset('r_heel', -0.1)  # 0.2

        # print('num phases: %d' % len(self.controller.phases))

    def collect_result(self):
        res = {}
        res['params'] = None if self.params is None else np.array(self.params)
        res['C'] = np.array(self.skel().C)
        # T = self.skel().COP
        # res['T'] = np.array(T) if T is not None else np.zeros(3)
        res['maxCy'] = max([C[1] for C in self.com_trajectory])
        res['t'] = self.world.t
        res['fallen'] = self.fallen
        # print 'result: ', res
        return res

    def reset_hook(self):
        self.fallen = False

    def terminated(self):
        C = self.skel().C
        Hy = self.skel().body('torso').C[1]

        if C[1] < 0.12 or math.fabs(C[0]) > 0.06 or \
           Hy < C[1]:  # May check |Cx| > 0.04
            self.fallen = True
            return True

        return (self.world.t > 1.7)
        # return (self.world.t > 3.0)  # For final production

    def __str__(self):
        res = self.collect_result()
        status = ""
        status += '[GPJump at %.4f' % self.world.t
        status += '(%d)' % self.controller.phase_index
        for key, value in self.collect_result().iteritems():
            if key in set(['params']):
                continue
            if hasattr(value, '__len__'):
                status += ' %s : %s' % (key, STR(value, 3))
            else:
                status += ' %s : %s' % (key, value)
        status += ' value = {'
        tasks = np.linspace(0.0, 1.0, 6)
        values = [self.evaluate(res, t) for t in tasks]
        status += ' '.join(['%.4f' % v for v in values])
        status += '}]'
        return status

    def __repr__(self):
        return 'problems.GPJump()'
