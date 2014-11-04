import time


class PrintTime(object):
    def __init__(self):
        self.time_records = []

    def notify_step(self, solver, model):
        self.time_records += [time.time()]

    def notify_solve(self, solver, model):
        self.time_records += [time.time()]
        total_time = self.time_records[-1] - self.time_records[0]
        print('total time: %.4fs' % total_time)
