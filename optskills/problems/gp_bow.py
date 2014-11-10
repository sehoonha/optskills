from sim_problem import SimProblem, PDController


class GPBow(SimProblem):
    def __init__(self):
        super(GPBow, self).__init__('urdf/BioloidGP/BioloidGP.URDF')
        self.init_state = self.skel().x
        self.init_state[0] = -0.47 * 3.14
        self.init_state[4] = 0.23
        self.init_state[5] = 0.23
        self.reset()
        self.controller = PDController(self.skel(), 60, 1.0, 0.5)
        self.controller.target = self.skel().q

    def simulate(self, sample):
        self.reset()
        self.set_params(sample)
        while(self.terminated()):
            self.step()
        return {}

    def set_params(self, params):
        pass

    def terminated(self):
        return (self.world.t > 0.5)
