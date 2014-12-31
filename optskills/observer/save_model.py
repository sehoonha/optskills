import json


class SaveModel(object):
    def __init__(self, _outputfile):
        self.outputfile = _outputfile

    def notify_init(self, solver, model):
        pass

    def notify_solve(self, solver, model):
        self.notify_step(solver, model)

    def notify_step(self, solver, model):
        print('-- Save model into %s' % self.outputfile)
        with open(self.outputfile, 'w+') as fp:
            data = {}
            data['prob'] = repr(solver.prob)
            data['mean_type'] = repr(model.mean_type)
            data['mean_params'] = repr(model.mean.params())
            json.dump(data, fp)
