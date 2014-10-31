#!/usr/bin/python

import problems
import solver
import observer

print 'Hello, OptSkills!'

prob = problems.Sphere()
mean_type = 'linear'
solver = solver.ParameterizedSolver(prob, mean_type)
solver.add_observer(observer.PlotValues())
solver.add_observer(observer.PlotMean(mean_type))

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
