from motormap import MotorMap
import re


def translate():
    inputfile = 'bio_gp_humanoid_kr.mtn'
    targets = ['f_r1', 'f_r_l', 'f_l_r']

    mmap = MotorMap()
    mmap.load('BioloidGPMotorMap.xml')
    targeted = False
    with open(inputfile) as fin:
        for line_no, line in enumerate(fin.readlines()):
            tokens = re.split('[ =]', line[:-1])
            tokens = [t.strip() for t in tokens if len(t) > 0]
            if len(tokens) == 0:
                continue
            cmd = tokens[0]
            if cmd == 'name':
                name = tokens[1] if len(tokens) > 1 else ''
                targeted = name in targets
                if targeted:
                    print()
                    print('name: %s' % tokens[1])
            elif cmd == 'step':
                if targeted:
                    mtv = [int(t) for t in tokens[1:-2]]
                    v = mmap.from_motor_pose(mtv)
                    print(repr(v))
            elif cmd == 'page_end':
                targeted = False
            # print line_no, tokens

if __name__ == '__main__':
    translate()
