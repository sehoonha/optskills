import numpy as np


NUMENTRIES = 26


class Step(object):
    def __init__(self, pose, pause, time):
        self.pose = pose
        while len(self.pose) < NUMENTRIES:
            self.pose += [512]

        self.pause = pause
        self.time = time

    def unit_pause(self):
        unit = 0.008
        return float(int(self.pause / unit)) * unit

    def unit_time(self):
        unit = 0.008
        return float(int(self.time / unit)) * unit

    def __str__(self):
        ret = "step="
        ret += " ".join("%d" % x for x in self.pose)
        ret += " %.3f %.3f" % (self.unit_pause(), self.unit_time())
        return ret


class Page(object):
    def __init__(self, name, compliance=None, next_page=0, exit_page=0,
                 repeat=1, speed_rate=1.0, ctrl_inertial=32):
        self.name = name
        self.compliance = compliance
        if self.compliance is None:
            self.compliance = [5] * NUMENTRIES
        self.steps = []

        self.next_page = next_page
        self.exit_page = exit_page
        self.repeat = repeat
        self.speed_rate = speed_rate
        self.ctrl_inertial = ctrl_inertial

    def play_param(self):
        return (self.next_page, self.exit_page, self.repeat,
                self.speed_rate, self.ctrl_inertial)

    def __str__(self):
        ret = "page_begin\n"
        ret += "name=%s\n" % self.name
        comp_tokens = " ".join(["%d" % x for x in self.compliance])
        ret += "compliance=%s\n" % comp_tokens
        ret += "play_param=%d %d %d %.1f %d\n" % self.play_param()
        for step in self.steps:
            ret += str(step) + '\n'
        ret += "page_end\n"
        return ret


class Motion(object):
    def __init__(self, _motormap):
        self.motormap = _motormap

        self.pages = []
        # pg = Page("Init")
        # pg.steps += [Step([512] * 19, 0.0, 1.0)]
        # self.pages += [pg]

    def header(self):
        ret = ""
        ret += "type=motion\n"
        ret += "version=1.01\n"
        ret += "enable=0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0\n"
        ret += "motor_type=" + "0 " * 25 + "0\n"
        return ret

    def add_page(self, name, poses, durations):
        if len(poses) != len(durations):
            print 'add_page Error!! lengths %d != %d' % \
                (len(poses), len(durations))

        pg = Page(name)
        for i, (q, t) in enumerate(zip(poses, durations)):
            print
            print 'Pose:', i
            motor_q = self.motormap.to_motor_pose(q).tolist()
            print 'Time:', t
            print 'Pose :', q.tolist()
            print 'Motor:', motor_q
            pg.steps += [Step(motor_q, 0.0, t)]
        print 'Add page OK'
        self.pages += [pg]
        return pg

    def fill_with_empty_pages(self):
        while len(self.pages) < 255:
            self.pages += [Page("")]

    def save(self, filename):
        with open(filename, 'w+') as fp:
            fp.write(chr(0xEF))
            fp.write(chr(0xBB))
            fp.write(chr(0xBF))
            fp.write(str(self))
        self.add_check_sum(filename)

    def add_check_sum(self, filename):
        file = open(filename, "r+")

        checksum = 0xEF + 0xBB + 0xBF + 1

        c = file.read(3)

        c = file.read(1)
        while not c == "":
            checksum = checksum + ord(c)
            c = file.read(1)

        checksum = -checksum & 0xFFFF
        firstbyte = checksum / 256
        secondbyte = checksum % 256
        print "byte: ", hex(firstbyte)
        print "byte: ", hex(secondbyte)

        file.write(chr(firstbyte))
        file.write(chr(secondbyte))

        file.close()

    def __str__(self):
        ret = self.header()
        for page in self.pages:
            ret += str(page)
        return ret
