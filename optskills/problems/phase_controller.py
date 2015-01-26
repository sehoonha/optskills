import numpy as np
from sim_problem import SPDController, PDController, JTController


class Phase(object):
    def __init__(self, _skel, _target, _terminate, _vfs=None):
        self.skel = _skel
        self.target = np.array(_target)
        self.terminate = _terminate
        self.vfs = _vfs if _vfs is not None else []

    def is_terminate(self, world, phase_begins):
        if isinstance(self.terminate, float):
            return (world.t - phase_begins >= self.terminate)
        elif self.terminate == 'no_contact':
            return self.has_no_contact()
        elif self.terminate == 'has_contact':
            return self.has_contact()
        else:
            return self.terminate(world)

    def has_no_contact(self):
        return len(self.skel.contacted_body_names()) == 0

    def has_contact(self):
        return len(self.skel.contacted_body_names()) > 0

    def set_target(self, dof, value):
        index = self.skel.dof_index(dof)
        self.target[index] = value

    def add_target_offset(self, dof, value):
        index = self.skel.dof_index(dof)
        self.target[index] += value

    def add_virtual_force(self, b, f):
        self.vfs += [(b, f)]


class PhaseController(object):
    def __init__(self, _world, _spd=False):
        self.world = _world
        self.spd = _spd
        self.loop_phase_index = None
        self.callback = None
        if self.spd:
            self.pd = SPDController(self.skel(), 300.0, 30.0, self.world.dt)
        else:
            # self.pd = PDController(self.skel(), 60.0, 3.0, 0.3)
            self.pd = PDController(self.skel(), 10.0, 0.3, 0.3)

        self.jt = JTController(self.skel())

        self.phases = []
        self.reset()

    def skel(self):
        return self.world.skels[-1]

    def phase(self, index=None):
        if index is not None:
            return self.phases[index]
        # If not, provide the current phase
        if self.phase_index < len(self.phases):
            return self.phases[self.phase_index]
        else:
            return None

    def clear_phases(self):
        self.phases = []

    def add_phase_from_now(self, duration):
        ph = Phase(self.skel(), self.skel().q, duration)
        self.phases += [ph]
        return ph

    def add_phase_from_prev(self, duration):
        target = np.array(self.phases[-1].target)
        ph = Phase(self.skel(), target, duration)
        self.phases += [ph]
        return ph

    def add_phase_from_pose(self, duration, target):
        ph = Phase(self.skel(), target, duration)
        self.phases += [ph]
        return ph

    def reset(self):
        self.phase_index = 0
        self.phase_begins = 0.0
        if self.phase() is not None:
            self.pd.target = self.phase().target

    def control(self):
        if self.phase().is_terminate(self.world, self.phase_begins):
            if self.phase_index + 1 < len(self.phases):
                self.phase_index += 1
            else:
                if self.loop_phase_index is not None:
                    self.phase_index = self.loop_phase_index
            self.phase_begins = self.world.t
            self.pd.target = self.phase().target

        # if self.world.t < 1.0:
        #     print self.world.t, self.phase_index, self.skel().C,
        #     print self.pd.target

        if self.callback is not None:
            self.callback()

        vf = np.zeros(self.skel().ndofs)
        for f in self.phase().vfs:
            bodies = f[0]
            force = f[1]
            vf += self.jt.control(bodies, force)
        return self.pd.control() + vf
