#!/usr/bin/python

import problems
import solver
import observer

print 'Hello, OptSkills!'

prob = problems.Sphere()
NUM_TASKS = 6
MEAN_TYPE = 'linear'
# solver = solver.ParameterizedSolver(prob, NUM_TASKS, MEAN_TYPE)
solver = solver.DirectSolver(prob, NUM_TASKS, MEAN_TYPE)
# solver.add_observer(observer.PlotValues())
# solver.add_observer(observer.PlotMean(mean_type))
solver.add_observer(observer.PrintTime())

print(solver)
res = solver.solve()
print('==== respond from solver ====')
print(res)
