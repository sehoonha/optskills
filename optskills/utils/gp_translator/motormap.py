import math
import numpy as np
from bs4 import BeautifulSoup


class Motor(object):
    def __init__(self, _index=-1, _id=-1, _name=""):
        self.index = _index
        self.id = _id
        self.name = _name

        self.min_angle = -5.0 / 6.0 * math.pi
        self.max_angle = 5.0 / 6.0 * math.pi

    def swap(self):
        (self.min_angle, self.max_angle) = (self.max_angle, self.min_angle)

    def add_offset(self, o):
        self.min_angle += o
        self.max_angle += o

    def from_motor(self, mv):
        return np.interp(mv, [0, 1024], [self.min_angle, self.max_angle])

    def to_motor(self, v):
        if self.min_angle < self.max_angle:
            ret = np.interp(v, [self.min_angle, self.max_angle], [0, 1024])
        else:
            ret = np.interp(v, [self.max_angle, self.min_angle], [1024, 0])
        print 'v: %.4f -> %d for %s' % (v, ret, str(self))
        return ret
        # return np.interp(v, [self.min_angle, self.max_angle], [0, 1024])

    def __str__(self):
        ret = "Motor(%d, id:%d, %s)" % (self.index, self.id, self.name)
        ret += "[(%.4f %.4f)]" % (self.min_angle, self.max_angle)
        return ret


class MotorMap(object):
    def __init__(self):
        self.nmotors = 0
        self.ndofs = 0
        self.motors = []

    def load(self, filename, ndofs=None):
        print 'MotorMap.load. filename:', filename
        with open(filename, 'r') as fp:
            soup = BeautifulSoup(fp)
            self.motors = [self.load_motor(t) for t in soup.find_all('motor')]
            for m in self.motors:
                print m

        self.nmotors = max([m.id for m in self.motors]) + 1
        self.ndofs = max([m.index for m in self.motors]) + 1
        if ndofs is not None:
            self.ndofs = ndofs
        print 'nmotors = ', self.nmotors
        print 'ndofs = ', self.ndofs

    def load_motor(self, tag):
        # tag = <motor ...> ... </motor>
        m = Motor(int(tag['index']), int(tag['id']), tag['name'])
        if tag.has_attr('swap') and int(tag['swap']) == 1:
            m.swap()
        if tag.has_attr('offset'):
            m.add_offset(float(tag['offset']))
        return m

    def to_motor_pose(self, v):
        mtv = np.array([512] * self.nmotors)
        for m in self.motors:
            mtv[m.id] = m.to_motor(v[m.index])
        return mtv

    def from_motor_pose(self, mtv):
        v = np.zeros(self.ndofs)
        for m in self.motors:
            v[m.index] = m.from_motor(mtv[m.id])
        return v
