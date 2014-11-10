#!/usr/bin/python

import problems
import solver
import observer

print 'Hello, OptSkills!'

NUM_TESTS = 11
NUM_TASKS = 6
MEAN_TYPE = 'linear'


def benchmark():
    obs_plot_values = observer.PlotValues()
    observers = [obs_plot_values, observer.PrintTime()]
    for i in range(2 * NUM_TESTS):
        # prob = problems.Sphere()
        prob = problems.MirroredSphere()
        if i % 2 == 0:
            s = solver.ParameterizedSolver(prob, NUM_TASKS, MEAN_TYPE)
        elif i % 2 == 1:
            s = solver.InterpolationSolver(prob, NUM_TASKS, MEAN_TYPE)
        else:
            s = solver.DirectSolver(prob, NUM_TASKS, MEAN_TYPE)
        for o in observers:
            s.add_observer(o)
        print(s)
        res = s.solve()
        print('==== respond from solver ====')
        print(res)

    obs_plot_values.plot()


def test_solver(name=None):
    obs_plot_values = observer.PlotValues()
    observers = [obs_plot_values, observer.PrintTime()]
    # prob = problems.Sphere()
    prob = problems.MirroredSphere()
    s = None
    if name == 'parameterized':
        s = solver.ParameterizedSolver(prob, NUM_TASKS, MEAN_TYPE)
        observers += [observer.PlotMean('linear')]
    elif name == 'direct':
        s = solver.DirectSolver(prob, NUM_TASKS, MEAN_TYPE)
    elif name == 'interpolation':
        s = solver.InterpolationSolver(prob, NUM_TASKS, MEAN_TYPE)
    else:
        return
    for o in observers:
        s.add_observer(o)
    print(s)
    res = s.solve()
    print('==== respond from solver ====')
    print(res)
    obs_plot_values.plot()

# benchmark()
test_solver('parameterized')
# test_solver('direct')
# test_solver('interpolation')
