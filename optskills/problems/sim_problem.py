import pydart
import numpy as np
from numpy.linalg import inv

DATA_PATH = '../optskills/pydart/data/'


def confine(x, lo, hi):
    return min(max(lo, x), hi)


def STR(vector, precision):
    if vector is None:
        return "[]"
    fmt = '%%.%df' % precision
    return "[" + ", ".join([fmt % v for v in vector]) + "]"


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


class SPDController:
    def __init__(self, _skel, _kp, _kd, _h):
        self.skel = _skel
        self.ndofs = self.skel.ndofs
        self.KP = np.diag([_kp] * self.ndofs)
        self.KD = np.diag([_kd] * self.ndofs)
        self.h = _h
        self.target = None
        self.step_counter = 0  # For debug

    def verbose(self):
        return False
        # return (self.step_counter % 100 == 0)

    def control(self):
        skel = self.skel

        invM = inv(skel.M + self.KD * self.h)
        p = -self.KP.dot(skel.q + skel.qdot * self.h - self.target)
        d = -self.KD.dot(skel.qdot)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.KD.dot(qddot) * self.h

        # Confine max torque
        tau[:6] = 0.0
        for i in range(6, self.ndofs):
            tau[i] = confine(tau[i], -600, 600)
        return tau


class JTController:
    """
    # Usage
    self.jt = JTController(self.skel)
    tau += self.jt.control( ["l_hand", "r_hand"], f )
    """
    def __init__(self, _skel):
        self.skel = _skel

    def control(self, bodynames, f):
        if not isinstance(bodynames, list):
            bodynames = [bodynames]

        tau = np.zeros(self.skel.ndofs)
        for bodyname in bodynames:
            # J = self.skel.getBodyNodeWorldLinearJacobian(bodyname)
            J = self.skel.body(bodyname).world_linear_jacobian()
            JT = np.transpose(J)
            tau += JT.dot(f)
        return tau


class SimProblem(object):
    world = None

    def __init__(self, skel_filename, make_ball=False):
        self.skel_filename = skel_filename
        self.ball = None
        self.ball_position = None
        if SimProblem.world is None:
            self.__init__pydart__(skel_filename, make_ball)
        self.world = SimProblem.world
        self.controller = None

    def __init__pydart__(self, skel_filename, make_ball):
        pydart.init()
        if '.skel' not in skel_filename:
            world = pydart.create_world(1.0 / 5000.0)
            world.add_skeleton(DATA_PATH + "sdf/ground.urdf", control=False)
            if make_ball:
                world.add_skeleton(DATA_PATH + "urdf/sphere.urdf",
                                   control=False)
                self.ball = world.skels[-1]
            world.add_skeleton(DATA_PATH + self.skel_filename)
        else:
            world = pydart.create_world(1.0 / 2000.0,
                                        DATA_PATH + skel_filename)
            world.skels[-1].set_joint_damping(0.15)
        SimProblem.world = world
        print('__init__pydart__ OK')

    def skel(self):
        return self.world.skels[-1]

    def set_init_state(self, dof, value):
        index = self.skel().dof_index(dof)
        self.init_state[index] = value

    def reset(self):
        self.skel().set_states(self.init_state)
        if self.ball is not None and self.ball_position is not None:
            q = np.zeros(6)
            q[3:] = self.ball_position
            self.ball.q = q
            self.ball.qdot = np.zeros(6)

        self.world.reset()
        self.com_trajectory = [self.skel().C]
        if hasattr(self.controller, 'reset'):
            self.controller.reset()

    def step(self):
        if self.controller is not None:
            tau = self.controller.control()
            tau[0:6] = 0.0
            self.skel().tau = tau
        self.world.step()
        self.com_trajectory += [self.skel().C]
        return self.terminated()
        # return False

    def terminated(self):
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
