import numpy as np
from numpy.linalg import norm
from sim_problem import SimProblem, SPDController, JTController, STR


class SimJumpController(object):
    def __init__(self, _world):
        self.world = _world
        self.spd = SPDController(self.skel(), 300.0, 30.0, self.world.dt)
        self.spd.target = self.skel().q
        self.jt = JTController(self.skel())

        self.dim = 5
        # self.dim = 3
        self.params = (np.random.rand(self.dim) - 0.5) * 2.0

        self.reset()
        # for i, dof in enumerate(self.skel().dofs):
        #     print i, dof.name
        # for i, body in enumerate(self.skel().bodies):
        #     print i, body.name

    def skel(self):
        return self.world.skels[-1]

    def reset(self):
        self.target_index = -1

        w = (self.params - (-1.0)) / 2.0  # Change to 0 - 1 Scale
        lo = np.array([-3.0, 0.0, -1.57, -300.0, -0.5])
        hi = np.array([3.0, -3.0, 1.57, 300.0, 0.5])
        # lo = np.array([-3.0, 0.0, -3.0])
        # hi = np.array([3.0, -3.0, 3.0])
        params = lo * (1 - w) + hi * w
        # print('self.params = %s' % self.params)
        # print('normalized params = %s' % params)
        (q0, q1, q2, f0, q3) = params
        # (q0, q1, q2) = params
        # f0 = 100.0

        # Set the first pose
        pose0 = self.skel().q
        pose0[6] = pose0[9] = q0  # Thighs
        pose0[14] = pose0[15] = q1  # Knees
        pose0[17] = pose0[19] = q2  # Heels
        pose0[28], pose0[31] = 0.5, -0.5  # Shoulder

        # Set the second pose
        pose1 = self.skel().q
        pose1[28], pose1[31] = -2.0, 2.0  # Shoulder
        pose1[29], pose1[32] = 0.5, -0.5  # Shoulder

        # Set the third pose
        pose2 = self.skel().q
        pose2[6] = pose2[9] = q3  # Thighs

        self.target_time = [0.0, 0.4, 0.8, 9.9e8]
        self.targets = [pose0, pose1, pose2]
        self.forces = [[],
                       [(["h_toe_left", "h_toe_right"], [0, -f0, 0])],
                       []]

    def control(self):
        next_t = self.target_time[self.target_index + 1]
        if self.world.t >= next_t:
            self.target_index += 1
            self.spd.target = self.targets[self.target_index]

        vf = np.zeros(self.skel().ndofs)
        for f in self.forces[self.target_index]:
            bodies = f[0]
            force = f[1]
            vf += self.jt.control(bodies, force)

        return self.spd.control() + vf


class SimJump(SimProblem):
    def __init__(self):
        super(SimJump, self).__init__('skel/fullbody1.skel')
        self.__init__simulation__()

        self.dim = self.controller.dim
        self.eval_counter = 0  # Well, increasing when simulated
        self.params = None

    def __init__simulation__(self):
        self.init_state = self.skel().x
        self.init_state[1] = -0.50 * 3.14
        self.init_state[4] = 0.88
        self.init_state[5] = 0.0

        self.reset()
        self.controller = SimJumpController(self.world)
        # self.controller = SPDController(self.skel(), 400.0, 40.0, h)
        # self.controller.target = self.skel().q

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
        C = np.array(result['C'])
        C[1] = result['maxCy']
        lo = np.array([0.0, 1.00, 0.0])
        hi = np.array([0.0, 1.50, 0.0])
        w = task
        C_hat = lo * (1 - w) + hi * w
        weight = np.array([0.2, 1.0, 0.2])
        obj = norm((C - C_hat) * weight) ** 2

        # Test height penalty
        height_penalty = 0.0 if result['C'][1] > 0.5 else 1.0
        obj += height_penalty

        # Calculate unbalanced penalty
        T = result['T']
        obj_balanced = 10.0
        if T is not None:
            weight = np.array([0.5, 0.0, 0.5])
            obj_balanced = norm((T - C) * weight) ** 2

        # Calculate parameter penalty
        params = result['params']
        penalty = 0.0
        if params is not None:
            for i in range(self.dim):
                v = params[i]
                penalty += max(0.0, v - 1.0) ** 2
                penalty += min(0.0, v - (-1.0)) ** 2
        return obj + obj_balanced + penalty

    def set_random_params(self):
        self.set_params(2.0 * (np.random.rand(self.dim) - 0.5))

    def set_params(self, x):
        self.params = x
        self.controller.params = x
        self.controller.reset()

    def collect_result(self):
        res = {}
        res['C'] = self.skel().C
        res['T'] = self.skel().COP
        res['maxCy'] = max([C[1] for C in self.com_trajectory])
        res['params'] = self.params
        return res

    def terminated(self):
        # contacted = self.skel().contacted_body_names()
        # print contacted
        # if self.world.t > 0.8:
        #     for body in contacted:
        #         if 'toe' in body:
        #             return True
        return (self.world.t > 1.3)

    def __str__(self):
        res = self.collect_result()
        status = ""
        status += '[SimJump at %.4f' % self.world.t
        for key, value in res.iteritems():
            if key == 'C':
                status += ' %s : %s' % (key, STR(value, 2))
            elif key == 'T':
                status += ' %s : %s' % (key, STR(value, 2))
            elif key == 'params':
                status += ' %s : %s' % (key, STR(value, 3))
            else:
                status += ' %s : %.4f' % (key, value)

        # Print Values
        status += ' value = {'
        tasks = np.linspace(0.0, 1.0, 6)
        values = [self.evaluate(res, t) for t in tasks]
        status += ' '.join(['%.4f' % v for v in values])
        status += '}'

        status += ']'
        return status

    def __repr__(self):
        return 'problems.SimJump()'
