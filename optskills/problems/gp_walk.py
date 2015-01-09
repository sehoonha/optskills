import math
import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, STR
import phase_controller
import gp_walk_poses


class GPWalk(SimProblem):
    def __init__(self):
        super(GPWalk, self).__init__('urdf/BioloidGP/BioloidGP.URDF',
                                     fps=2000.0)
        self.__init__simulation__()

        self.dim = 7
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

        self.set_params(np.array([0.14, 0.3, -0.15, -0.1, -0.2, 0.0, 0.0]))
        self.controller.reset()

    def balance(self):
        Cd = self.skel().Cdot
        bal = Cd[2]
        bal2 = Cd[0]

        l0 = self.skel().dof_index('l_thigh')
        r0 = self.skel().dof_index('r_thigh')
        l1 = self.skel().dof_index('l_shin')
        r1 = self.skel().dof_index('r_shin')
        l2 = self.skel().dof_index('l_heel')
        r2 = self.skel().dof_index('r_heel')

        x0 = self.skel().dof_index('l_hip')
        y0 = self.skel().dof_index('r_hip')
        x1 = self.skel().dof_index('l_foot')
        y1 = self.skel().dof_index('r_foot')

        qhat = np.array(self.controller.phase().target)
        bal *= 1.0
        # print('t: %0.4f bal: %.6f' % (self.world.t, bal))
        qhat[l0] -= bal * 1.0
        qhat[r0] -= bal * 1.0
        qhat[l1] -= bal * 1.0
        qhat[r1] -= bal * 1.0
        qhat[l2] -= bal * 1.0
        qhat[r2] -= bal * 1.0

        bal2 *= (0.2)
        qhat[x0] += bal2 * 1.0
        qhat[y0] += bal2 * 1.0
        qhat[x1] += bal2 * 1.0
        qhat[y1] += bal2 * 1.0

        self.controller.pd.target = qhat

    def simulate(self, sample):
        self.eval_counter += 1

        self.reset()
        self.set_params(sample)
        self.reset()
        while not self.terminated():
            self.step()
        # print 'result:', self.params, self.collect_result()
        return self.collect_result()

    def evaluate(self, result, task):
        w = task
        # Calculate the validity of C
        C = result['C']
        lo = np.array([0.0, 0.16, 0.20])
        hi = np.array([0.0, 0.16, 0.40])
        C_hat = lo * (1 - w) + hi * w
        weight = np.array([0.1, 0.0, 3.0])
        obj = norm((C - C_hat) * weight) ** 2

        # Time penalty
        t = result['t']
        t_penalty = 0.0
        if result['f']:
            t_penalty = 0.5 + (0.1 * (5.0 - t)) ** 2

        # Calculate parameter penalty
        params = result['params']
        penalty = 0.0
        if params is not None:
            for i in range(self.dim):
                v = params[i]
                penalty += max(0.0, v - 1.0) ** 2
                penalty += min(0.0, v - (-1.0)) ** 2

        return 10.0 * (obj + t_penalty + penalty)

    def set_random_params(self):
        pass

    def set_params(self, x):
        self.params = x
        w = (x - (-1.0)) / 2.0  # Change to 0 - 1 Scale
        # lo = np.array([0.0, -1.0, -0.5, -0.5, -0.5, -0.5])
        # hi = np.array([0.2, 1.0, 0.5, 0.5, 0.5, 0.5])
        lo = np.array([0.0, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2])
        hi = np.array([0.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
        params = lo * (1 - w) + hi * w
        (t0, q1, q2, q3, q4, q5, q6) = params
        # print t0, q1, q2, q3, q4

        self.reset()
        self.controller.clear_phases()

        for t, q in zip(gp_walk_poses.durations, gp_walk_poses.targets):
            phase_index = len(self.controller.phases)
            if phase_index == 4 or phase_index == 7:
                t = 0.072 + t0  # 0.168
            ph = self.controller.add_phase_from_pose(t, q)

            if phase_index in set([3, 4, 5]):
                if phase_index == 4:
                    ph.add_target_offset('r_thigh', q1)  # Swing hip. 1.00
                    ph.add_target_offset('r_shin', q5)  # Swing knee. 0.00
                    ph.add_target_offset('l_heel', q2)  # Stand ankle. 0.20
                if phase_index == 5:
                    ph.add_target_offset('r_hip', q3)  # Swing hip. 0.10
                    ph.add_target_offset('r_thigh', q6)  # Swing hip. 0.0
                    ph.add_target_offset('r_heel', q4)  # Stand ankle. -0.15

            elif phase_index in set([6, 7, 8]):
                if phase_index == 7:
                    ph.add_target_offset('l_thigh', q1)  # Swing hip. 1.00
                    ph.add_target_offset('l_shin', q5)  # Swing knee. 0.00
                    ph.add_target_offset('r_heel', q2)  # Stand ankle. 0.20
                if phase_index == 8:
                    ph.add_target_offset('l_hip', -q3)  # Swing ankle -0.10
                    ph.add_target_offset('l_thigh', q6)  # Swing hip. 0.0
                    ph.add_target_offset('l_heel', q4)  # Stand ankle -0.15

    def collect_result(self):
        res = {}
        res['t'] = self.world.t
        res['f'] = self.is_fallen()
        res['C'] = self.skel().C
        res['params'] = self.params
        return res

    def is_fallen(self):
        if self.world.t < 0.1:
            return False
        contacted = self.skel().contacted_body_names()
        C = self.skel().C
        if math.fabs(C[0]) > 0.15:
            return True
        if set(contacted) - set(['l_foot', 'r_foot', 'l_shin',
                                 'r_shin', 'l_heel', 'r_heel']):
            return True
        return False

    def terminated(self):
        return self.is_fallen() or (self.world.t > 3.0)

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
        # values = [self.evaluate(res, 0.0)]
        status += ' '.join(['%.6f' % v for v in values])
        status += '}]'
        return status

    def __repr__(self):
        return 'problems.GPWalk()'
