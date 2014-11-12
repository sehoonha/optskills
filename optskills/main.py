#!/usr/bin/python

import problems
import solver
import observer

print 'Hello, OptSkills!'

NUM_TESTS = 11
NUM_TASKS = 6
MEAN_TYPE = 'linear'
PROBLEM_CODE = None


def save(prob, model, filename):
    import json
    with open(filename, 'w+') as fp:
        data = {}
        data['prob'] = repr(prob)
        data['mean_type'] = repr(model.mean_type)
        data['mean_params'] = repr(model.mean.params())
        json.dump(data, fp)


def benchmark():
    obs_plot_values = observer.PlotValues()
    observers = [obs_plot_values, observer.PrintTime()]
    for i in range(2 * NUM_TESTS):
        # prob = problems.Sphere()
        prob = problems.MirroredSphere()
        # prob = problems.GPBow()
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


def create_problem():
    return eval(PROBLEM_CODE)


def create_solver(solver_name, prob):
    if solver_name == 'parameterized':
        return solver.ParameterizedSolver(prob, NUM_TASKS, MEAN_TYPE)
    elif solver_name == 'interpolation':
        return solver.InterpolationSolver(prob, NUM_TASKS, MEAN_TYPE)
    elif solver_name == 'direct':
        return solver.DirectSolver(prob, NUM_TASKS, MEAN_TYPE)
    else:
        return None


def evaluate(name):
    obs_plot_values = observer.PlotValues()
    observers = [obs_plot_values, observer.PrintTime()]
    prob = create_problem()
    s = create_solver(name, prob)
    if name == 'parameterized':
        observers += [observer.PlotMean('linear')]
    for o in observers:
        s.add_observer(o)
    print(s)
    res = s.solve()
    print('==== respond from solver ====')
    print(res)
    obs_plot_values.plot()
    save(prob, s.model, 'result_%s.json' % name)


def mpi_evaluate(solver_name):
    import os
    pid = os.getpid()
    print('==== begin solver: %d ====' % pid)
    obs_plot_values = observer.PlotValues()
    observers = [obs_plot_values, observer.PrintTime()]
    # prob = problems.Sphere()
    prob = problems.MirroredSphere()
    s = create_solver(solver_name, prob)
    for o in observers:
        s.add_observer(o)
    res = s.solve()
    print('==== respond from solver %d ====' % pid)
    print(res)
    return (pid, solver_name, obs_plot_values.data)


def mpi_benchmark(solvers, NUM_CORES=4):
    # obs_plot_values = observer.PlotValues()
    import multiprocessing as mp
    import time
    begin_time = time.time()
    print ('-' * 80)
    print('all solvers:')
    print('%s' % solvers)
    print ('-' * 80)
    pool = mp.Pool(NUM_CORES)
    results = pool.map(mpi_evaluate, solvers)
    print ('\n\n')
    print ('-' * 80)
    collected_data = {}
    for i, res in enumerate(results):
        (pid, solver_name, solver_data) = res
        print i, pid, solver_data
        # Merge solver data into one structure
        for name, exp_list in solver_data.iteritems():
            if name not in collected_data:
                collected_data[name] = []
            collected_data[name] += exp_list
    print('-' * 80)
    print('collected data: %s' % collected_data)
    print ('-' * 80)
    print ('plot...')
    pl = observer.PlotValues()
    pl.data = collected_data
    pl.plot()
    print ('plot... done')
    end_time = time.time()
    print ('total %.4fs elapsed' % (end_time - begin_time))

# PROBLEM_CODE = 'problems.Sphere()'
PROBLEM_CODE = 'problems.MirroredSphere()'
# PROBLEM_CODE = 'problems.GPBow()'

# evaluate('parameterized')
mpi_benchmark(['parameterized'] * 51)
# mpi_benchmark(['parameterized', 'direct'] * 11)
# mpi_benchmark(['parameterized', 'direct', 'interpolation'] * 51)
