import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, STR
import phase_controller


class GPKick(SimProblem):
    def __init__(self):
        super(GPKick, self).__init__('urdf/BioloidGP/BioloidGP.URDF',
                                     make_ball=True)
        self.ball_position = np.array([0.03, 0.03, 0.10])
        r = 0.04
        m = 0.05
        I = (2.0 * m * r * r) / 3.0
        print('I = %.8f' % I)
        self.__init__simulation__()

        self.dim = 5
        self.eval_counter = 0  # Well, increasing when simulated
        self.params = None

    def __init__simulation__(self):
        self.init_state = self.skel().x
        self.init_state[0] = -0.50 * 3.14
        self.init_state[4] = 0.230
        self.init_state[5] = 0.230
        self.reset()
        self.controller = phase_controller.PhaseController(self.world)

        self.set_params(np.array([0.14, 0.3, -0.15, -0.1, -0.2]))
        self.controller.reset()

    def simulate(self, sample):
        self.eval_counter += 1

        self.reset()
        self.set_params(sample)
        while not self.terminated():
            self.kick()
        # print 'result:', self.params, self.collect_result()
        return self.collect_result()

    def evaluate(self, result, task):
        w = task
        # Calculate the validity of P (swing foot location)
        B = result['B']
        lo = np.array([0.025, 0.04, 0.30])
        hi = np.array([0.025, 0.04, 0.70])
        B_hat = lo * (1 - w) + hi * w
        weight = np.array([1.0, 1.0, 2.0]) * 2.0
        obj = norm((B - B_hat) * weight) ** 2

        # Calculate the balance penaly
        Cx = result['C'][0] - (-0.009)
        Cz = result['C'][2] - (0.014)
        b_penalty = (Cx * 5.0) ** 2 + (Cz * 5.0) ** 2

        # Time penalty
        t = result['t']
        t_penalty = 0.0
        if t < 2.5:
            t_penalty = (2.5 - t)

        # Calculate parameter penalty
        params = result['params']
        penalty = 0.0
        if params is not None:
            for i in range(self.dim):
                v = params[i]
                penalty += max(0.0, v - 1.0) ** 2
                penalty += min(0.0, v - (-1.0)) ** 2

        return obj + b_penalty + t_penalty + penalty

    def set_random_params(self):
        pass

    def set_params(self, x):
        self.params = x
        w = (x - (-1.0)) / 2.0  # Change to 0 - 1 Scale
        lo = np.array([-0.2, -1.57, -1.57, 0.0, -1.57])
        hi = np.array([0.2, 0.0, 0.0, 1.57, 0.0])
        params = lo * (1 - w) + hi * w
        (q0, q1, q2, q3, q4) = params
        print 'q:', q0, q1, q2, q3, q4

        self.reset()
        self.controller.clear_phases()
        # The first phase
        phase = self.controller.add_phase_from_now(0.5)
        phase.set_target('r_hip', -q0)  # -0.16
        phase.set_target('r_foot', q0)  # 0.16

        # The second phase
        phase = self.controller.add_phase_from_prev(0.3)
        phase.set_target('l_thigh', q1)  # -0.8
        phase.set_target('l_shin', q2)  # -1.0
        phase.set_target('l_heel', -0.5 * q2)  # 0.5

        # The third phase
        phase = self.controller.add_phase_from_prev(0.3)
        phase.set_target('l_thigh', q3)  # 0.55 ~ 1.0
        phase.set_target('l_shin', q4)  # -0.5
        phase.set_target('l_heel', -0.5 * q4)  # 0.3
        phase.set_target('r_heel', -0.05)  # -0.04
        # print('num phases: %d' % len(self.controller.phases))

    def collect_result(self):
        res = {}
        res['t'] = self.world.t
        res['C'] = self.skel().C
        res['B'] = self.ball.C
        res['dB'] = self.ball.Cdot
        res['params'] = self.params
        return res

    def terminated(self):
        contacted = self.skel().contacted_body_names()
        for b in ['torso', 'l_hand', 'r_hand']:
            if b in contacted:
                return True
        dB = self.ball.Cdot
        if dB[2] < 0.0 and self.world.t > 1.5:
            return True
        return (self.world.t > 5.0)

    def __str__(self):
        res = self.collect_result()
        status = ""
        status += '[GPKick at %.4f' % self.world.t
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
        return 'problems.GPKick()'
