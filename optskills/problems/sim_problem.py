import pydart
import numpy as np

DATA_PATH = '../optskills/pydart/data/'


def confine(x, lo, hi):
    return min(max(lo, x), hi)


class PDController:
    def __init__(self, _skel, _kp, _kd, _effort_ratio=1.0):
        self.skel = _skel
        self.ndofs = self.skel.ndofs
        self.kp = np.array([_kp] * self.ndofs)
        self.kd = np.array([_kd] * self.ndofs)
        self.target = None
        self.effort_ratio = _effort_ratio
        self.step_counter = 0  # For debug

    def verbose(self):
        return False
        # return (self.step_counter % 100 == 0)

    def control(self):
        q = self.skel.q
        qdot = self.skel.qdot

        tau = np.zeros(self.ndofs)
        tau_lo = self.skel.tau_lo * self.effort_ratio
        tau_hi = self.skel.tau_hi * self.effort_ratio

        for i in range(6, self.ndofs):
            tau[i] = -self.kp[i] * (q[i] - self.target[i]) \
                     - self.kd[i] * qdot[i]
            # tau[i] = confine(tau[i], -self.maxTorque, self.maxTorque)
            tau[i] = confine(tau[i], tau_lo[i], tau_hi[i])
            # tau[i] = confine(tau[i], -100.0, 100.0)  # Ugly..
        self.step_counter += 1
        return tau


class SimProblem(object):
    world = None

    def __init__(self, skel_filename):
        self.eval_counter = 0  # Well, increasing when simulated
        self.skel_filename = skel_filename
        if SimProblem.world is None:
            self.__init__pydart__(skel_filename)
        self.world = SimProblem.world
        self.controller = None

    def __init__pydart__(self, skel_filename):
        pydart.init()
        world = pydart.create_world(1.0 / 2000.0)
        world.add_skeleton(DATA_PATH + "sdf/ground.urdf", control=False)
        world.add_skeleton(DATA_PATH + self.skel_filename)
        world.skels[-1].set_joint_damping(0.15)
        SimProblem.world = world
        print('__init__pydart__ OK')

    def skel(self):
        return self.world.skels[-1]

    def reset(self):
        self.skel().set_states(self.init_state)
        self.world.reset()

    def step(self):
        if self.controller is not None:
            tau = self.controller.control()
            tau[0:6] = 0.0
            self.skel().tau = tau
        self.world.step()
        return False

    def render(self):
        self.world.render()

    def simulate(self, sample):
        self.eval_counter += 1
        return 'Simulated'

    def evaluate(self, result, task):
        return 0.0

    def __str__(self):
        return "[SimProblem at %.4f]" % self.world.t
