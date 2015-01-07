#!/usr/bin/python

import model
import problems
import numpy as np
from sample import Sample

print 'Hello, OptSkills!'

NUM_TESTS = 11
NUM_TASKS = 6
NUM_TEST_TASKS = 51
MEAN_TYPE = 'linear'
PROBLEM_CODE = None

prob = problems.SimJump()
print('Problem = %s' % prob)


def plot_results():
    print('Start plotting...')
    import matplotlib.pyplot as plt
    x = np.linspace(0.0, 1.0, NUM_TEST_TASKS)
    data_our = [0.0020691989731046034, 0.0018802702703689003, 0.0014240991898102067, 0.0010977461606589273, 0.00091503805129981559, 0.00084385953410101133, 0.0007222334603555592, 0.00070347811415337653, 0.00056376555648974625, 0.00046212615881941719, 0.00031641675535694851, 0.00026219346586970401, 0.00023613125206528885, 0.00018644566128347761, 0.00042990216189321484, 0.00012931532082256661, 0.0002977702828869678, 0.00018066929556262014, 0.00032058600986565412, 0.00041943514403973973, 0.00049189169645863268, 0.00056285804687812005, 0.0007538207081152794, 0.0010561016884332883, 0.0013256488807460148, 0.0013492802384026201, 0.001665776855810085, 0.0020266474670462033, 0.0014960357202210034, 0.0018400967738305279, 0.0025172559007519907, 0.0029099174299356727, 0.0027716843497072245, 0.0025649015396122209, 0.0040959101275817901, 0.0044989614918560673, 0.0042540275875463072, 0.004330954006659081, 0.0056496610635377769, 0.0051403847559327517, 0.0056530120309980798, 0.0059672718721249969, 0.006163797742034482, 0.0073592267859982482, 0.0070352395950811835, 0.0068190679862202173, 0.0072149749465441034, 0.0074366225524383784, 0.0068709166619021472, 0.0068204407317632876, 0.0062396335324729924]
    data_ind = [0.0014265832284794575, 0.011235763848433321, 0.12125185426500416, 1.1500348821834878, 0.13009005708390795, 0.082118749233913019, 0.020339612325519583, 0.024543243437702884, 0.01492153830144723, 0.0024608795563587655, 0.00065995220709045579, 0.001564849755606438, 0.012393790720140496, 0.052781229288267638, 0.022817337426808384, 0.018465261539972608, 0.049009008606099982, 1.1337313016904675, 1.1439588765129405, 0.067959310389766672, 0.00081289484171559727, 0.049908751914105758, 2.1006266718499775, 1.1501535384201151, 0.096893984005530828, 0.037522519253373132, 0.024475952479590488, 0.037423528233820967, 0.037561719688885303, 0.0064155268526826141, 0.0013058284707900001, 0.00080354245884429727, 0.00351741135595642, 0.0068643146427746094, 0.0095720940940730893, 0.012762864035722779, 0.022857994989205408, 0.014137758076673921, 0.0092798732016238054, 0.0055869640942253496, 0.0023227686386581244, 0.0052600456886403068, 0.0044519979507840661, 0.004087679356100507, 0.0034941863964245978, 0.0029345670174764516, 0.0022863055561975214, 0.0015427709992062338, 0.0014445297237211038, 0.00095027228564662598, 0.00088609844220362689]
    data_high = [0.00088787860715004783, 0.00055761052562195573, 0.00094626001160654383, 0.0001757412739468282, 0.0016602416911492102, 0.0071938105321318899, 0.00098133872636892421, 0.00083572066524677192, 0.00089242614913867893, 0.00043025612765510118, 0.00044394967468933794, 0.00029430239595325247, 0.00036913725473343201, 0.000633611114532656, 0.00085239048977760114, 0.00068012791867127402, 0.00092987396154559413, 0.00029547139988714857, 0.00084603265254383168, 0.00087964694869285581, 0.0041006689252678389, 0.00051305334613264302, 0.00031906482024836118, 0.0009791156653515896, 0.00043576151963676738, 0.00062288788405916087, 0.0044224435725845916, 0.0010898673126283687, 0.00074682333396133035, 0.00037780079254310078, 0.0045350522471827009, 0.0037682566486465685, 0.0013170406913670743, 0.0043153087001235211, 0.00093570080057016282, 0.0030148127871983833, 0.00093613954126076104, 0.022574077255323195, 0.00074461956341457722, 0.00095902243009157923, 0.00028916433963021155, 0.0014286279438400805, 0.00098109869967655727, 0.0062343791070488038, 0.0009817386572504895, 0.0006767961162165337, 0.0010907634024003442, 0.0007678393426329012, 0.0020862895183892103, 0.00067888024928055733, 0.00082598105266723813]
    plt.ioff()
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    plt.plot(x, data_our, 'r', linewidth=2)
    plt.plot(x, data_ind, 'g', linewidth=2)
    plt.plot(x, data_high, 'b', linewidth=2)
    plt.vlines([0.2, 0.4, 0.6, 0.8], 0.0, 10.0, linestyles='dashed')
    font = {'size': 28}
    plt.xlabel('Task', fontdict=font)
    plt.ylabel('Cost', fontdict=font)
    plt.axes().set_yscale('log')
    plt.tick_params(axis='x', labelsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.legend(['Ours', 'Individual', 'Individual (High-res)'],
               numpoints=1, fontsize=26)
    plt.savefig('plot_vs_ind.png')
    plt.close(fig)
    print('Done plotting.')
    exit(0)
plot_results()


# Case 1. If we use individual approach..
mean = model.mean.Interpolation(NUM_TASKS)

# # Case 2. If we use our algorithm..
# import json
# with open('result_parameterized_04.json') as fp:
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
    if v > 10.0:
        v = v / 10 + 1.0
    print('%.4f, pt = %s , value = %.6f' % (w, repr(pt), v))
    values += [v]
print('average: %.8f' % np.mean(values))
print('max: %.8f' % np.max(values))
print('values: %s' % values)
