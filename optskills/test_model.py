#!/usr/bin/python

import model
import problems
import numpy as np
from sample import Sample

print 'Hello, OptSkills!'

NUM_TESTS = 11
NUM_TASKS = 6
NUM_TEST_TASKS = 21
MEAN_TYPE = 'linear'
PROBLEM_CODE = None

prob = problems.SimJump()
print('Problem = %s' % prob)

# Case 1. If we use individual approach..
mean = model.mean.Interpolation(NUM_TASKS)

# # Case 2. If we use our algorithm..
# import json
# with open('result_parameterized_03.json') as fp:
#     data = json.load(fp)
#     my_params = eval('np.' + data['mean_params'])
# tasks = np.linspace(0.0, 1.0, 6)
# my_model = model.Model(prob.dim, tasks, 'linear')
# my_model.mean.set_params(my_params)
# mean = my_model.mean

# resume all the cases..
print('mean = %s' % mean)

test_tasks = np.linspace(0.0, 1.0, NUM_TEST_TASKS)
values = []
for w in test_tasks:
    pt = mean.point(w)
    s = Sample(pt, prob)
    v = s.evaluate(w)
    print('%.4f, pt = %s , value = %.6f' % (w, repr(pt), v))
    values += [v]
print('average: %.8f' % np.mean(values))
print('max: %.8f' % np.max(values))
