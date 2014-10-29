#!/usr/bin/python
from sample import Sample
import numpy as np
import problems

print 'Hello, OptSkills!'

prob = problems.Sphere()

s = Sample([0.4, 0.4], prob)
print('Sample s %s ' % s)
for task in np.linspace(0.0, 1.0, 6):
    print('%s : %s' % (task, s.evaluate(task)))
