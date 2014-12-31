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


def create_problem():
    return eval(PROBLEM_CODE)


def create_solver(solver_name, prob):
    print('create_solver: [%s]' % solver_name)
    if solver_name == 'parameterized':
        return solver.ParameterizedSolver(prob, NUM_TASKS, MEAN_TYPE)
    elif solver_name == 'parameterized_cubic':
        return solver.ParameterizedSolver(prob, NUM_TASKS, 'cubic')
    elif solver_name == 'interpolation':
        return solver.InterpolationSolver(prob, NUM_TASKS, MEAN_TYPE)
    elif solver_name == 'direct':
        return solver.DirectSolver(prob, NUM_TASKS, MEAN_TYPE)
    elif solver_name == 'sampler':
        return solver.Sampler(prob, NUM_TASKS, MEAN_TYPE)
    elif 'parameterized|' in solver_name:
        alg = solver_name.split('|')[1]
        print('create_solver: alg = %s' % alg)
        return solver.ParameterizedSolver(prob, NUM_TASKS, MEAN_TYPE, alg)
    else:
        return None


def evaluate(name, plotting=True):
    import os
    obs_plot_values = observer.PlotValues('data_%s.csv' % name)
    observers = [obs_plot_values, observer.PrintTime()]
    observers += [observer.SaveModel('result_%s.json' % name)]
    prob = create_problem()
    s = create_solver(name, prob)
    # if name == 'parameterized':
    #     observers += [observer.PlotMean('linear')]
    for o in observers:
        s.add_observer(o)
    print(s)
    res = s.solve()
    print('==== respond from solver ====')
    print(res)
    if plotting:
        obs_plot_values.plot(PROBLEM_CODE)
    if hasattr(s, 'model'):
        save(prob, s.model, 'result_%s.json' % name)
    pid = os.getpid()
    return (pid, name, obs_plot_values.data)


def benchmark(solvers):
    # obs_plot_values = observer.PlotValues()
    import time
    begin_time = time.time()
    print ('-' * 80)
    print('all solvers:')
    print('%s' % solvers)
    print ('-' * 80)

    results = [evaluate(s, False) for s in solvers]
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
    pl.save('data_benchmark.csv')
    pl.plot(PROBLEM_CODE)
    print ('plot... done')
    end_time = time.time()
    print ('total %.4fs elapsed' % (end_time - begin_time))


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
    pl.plot(PROBLEM_CODE)
    print ('plot... done')
    end_time = time.time()
    print ('total %.4fs elapsed' % (end_time - begin_time))


def plot(filename):
    print('plot [%s]' % filename)
    obs_plot_values = observer.PlotValues()
    obs_plot_values.load(filename)
    obs_plot_values.plot(PROBLEM_CODE)


def copy_and_replot(expname):
    import shutil
    csv_filename = '%s.csv' % expname
    png_filename = '%s.png' % expname
    shutil.copy('data_benchmark.csv', csv_filename)
    print ('copy data into %s' % csv_filename)
    plot(csv_filename)
    print ('plot data from %s' % csv_filename)
    shutil.copy('plot_values.png', png_filename)
    print ('copy plot into %s' % png_filename)
    print ('done!')

# PROBLEM_CODE = 'problems.Sphere()'
# PROBLEM_CODE = 'problems.Sphere(_seg_type="cubic")'
# PROBLEM_CODE = 'problems.MirroredSphere()'
# PROBLEM_CODE = 'problems.GPBow()'
# PROBLEM_CODE = 'problems.GPStep()'
# PROBLEM_CODE = 'problems.GPKick()'
# PROBLEM_CODE = 'problems.GPWalk()'
PROBLEM_CODE = 'problems.SimJump()'
# PROBLEM_CODE = 'problems.CEC15(2, "bent_cigar")'
# seg = "[[-0.5, -0.1], [0.0, 0.1], [0.5, -0.1]]"
# adjust = "[0.5, 1.0]"
# PROBLEM_CODE = 'problems.CEC15(2, "bent_cigar", %s, "quadratic", 0.5, %s)' \
#                % (seg, adjust)
# PROBLEM_CODE = 'problems.CEC15(2, "weierstrass")'

# seg = "[[-0.5, -0.1], [0.0, 0.1], [0.5, -0.1]]"
# adjust = "[0.5, 1.5]"
# PROBLEM_CODE = 'problems.CEC15(2, "weierstrass", %s, "quad", 0.01, %s)' \
#                % (seg, adjust)

# seg = "[[-0.5, -0.1], [-0.4, 0.1]]"
# # PROBLEM_CODE = 'problems.CEC15(2, "weierstrass", %s, "linear", 1.0)' % seg
# PROBLEM_CODE = 'problems.CEC15(2, "schwefel", %s, "linear", 1.0)' % seg

# PROBLEM_CODE = 'problems.CEC15(2, "schwefel")'
# MEAN_TYPE = 'cubic'

math_problems = []
math_problems += [('problems.Sphere()', 'sphere')]
math_problems += [('problems.MirroredSphere()', 'mirrored')]
seg = "[[-0.5, -0.1], [-0.4, 0.1]]"
math_problems += [('problems.CEC15(2, "weierstrass", %s, "linear", 1.0)' % seg,
                   'weierstrass')]
math_problems += [('problems.CEC15(2, "schwefel", %s, "linear", 1.0)' % seg,
                   'schwefel')]


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'parameterized':
            evaluate('parameterized')
        elif cmd == 'parameterized2':
            evaluate('parameterized|cov_rank_1')
        elif cmd == 'direct':
            evaluate('direct')
        elif cmd == 'interpolation':
            evaluate('interpolation')
        elif cmd == 'benchmark':
            times = 11 if len(sys.argv) == 2 else int(sys.argv[2])
            print('Command = %s Times = %d' % (cmd, times))
            benchmark(['parameterized', 'direct'] * times)

        elif cmd == 'plot':
            filename = sys.argv[2]
            plot(filename)
        elif cmd == 'sampling':
            evaluate('sampler', False)
        exit(0)
    # evaluate('parameterized')
    # evaluate('parameterized_cubic')
    # evaluate('direct')
    # evaluate('interpolation')
    # evaluate('sampler', False)
    # evaluate('parameterized|mean_best,step_1_5,cov_rank_1')
    # evaluate('parameterized|mean_all,step_1_5,cov_all')
    # evaluate('parameterized|mean_rand,step_success,cov_rank_1')
    # mpi_benchmark(['parameterized'] * 11)
    # mpi_benchmark(['parameterized', 'direct'] * 21)
    # benchmark(['parameterized', 'direct'] * 5)
    # mpi_benchmark(['parameterized', 'interpolation'] * 5)
    # mpi_benchmark(['parameterized', 'direct', 'interpolation'] * 3, 1)
    # benchmark(['parameterized'] * 11)
    # benchmark(['parameterized', 'direct'] * 11)
    # benchmark(['parameterized', 'parameterized_cubic'] * 11)
    # benchmark(['parameterized|step_1_5',
    #            'parameterized|step_success',
    #            'direct'] * 21)
    # copy_and_replot('stepalgs02_weierstrass')

    print('sleep 1 seconds..')
    import time
    time.sleep(1)

    # # A full benchmarks for covariance matrix algorithms
    # for i in range(len(math_problems)):
    #     PROBLEM_CODE, shortname = math_problems[i]
    #     print('start %s' % shortname)
    #     # benchmark(['direct', 'parameterized|cov_rank_1'])
    #     benchmark(['parameterized|cov_rank_1',
    #                'parameterized|cov_all',
    #                'direct'] * 21)
    #     print('done with %s' % shortname)
    #     print('sleep 1 seconds..')
    #     time.sleep(1)
    #     copy_and_replot('covariance_prob%02d_%s' % (i, shortname))
    #     print('sleep 1 seconds..')
    #     time.sleep(1)

    # A full benchmarks for all algorithms
    for i in range(len(math_problems)):
        if i != 3:
            continue
        PROBLEM_CODE, shortname = math_problems[i]
        print('start %s' % shortname)
        benchmark(['parameterized|draw_uniform,cov_rank1',
                   'parameterized|draw_uniform,cov_all',
                   'direct'] * 31)
        print('done with %s' % shortname)
        print('sleep 1 seconds..')
        time.sleep(1)
        copy_and_replot('cov_uniform_prob%02d_%s' % (i, shortname))
        print('sleep 1 seconds..')
        time.sleep(1)
