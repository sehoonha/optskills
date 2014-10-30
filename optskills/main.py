#!/usr/bin/python

import problems
import solver

print 'Hello, OptSkills!'

prob = problems.Sphere()
solver = solver.ParameterizedSolver(prob)

print(solver)
res = solver.solve()
print('==== respond from solver ====')
print(res)

# s = Sample([0.4, 0.4], prob)
# tasks = np.linspace(0.0, 1.0, 6)
# print('Sample s: %s ' % s)

# for task in tasks:
#     print('%s : %s' % (task, s.evaluate(task)))

# model = model.Model(prob.dim, tasks, 'linear')
# print('Model: %s' % model)
