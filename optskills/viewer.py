print 'Hello Pydart'

import sys
import os
import signal


def signal_handler(signal, frame):
    print 'You pressed Ctrl+C! Bye.'
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

from OpenGL.GLUT import *
from PyQt4 import QtGui
from PyQt4 import QtCore
from glwidget import GLWidget
from pydart import Trackball
import numpy as np
import problems
import model


class MyWindow(QtGui.QMainWindow):
    def __init__(self, filename=None):
        super(MyWindow, self).__init__()

        # Check and create captures directory
        if not os.path.isdir('captures'):
            os.makedirs('captures')

        self.tasks = np.linspace(0.0, 1.0, 6)

        # Create a simulation
        if filename is None:
            # self.prob = problems.SimJump()
            # self.prob = problems.GPBow()
            # self.prob = problems.GPStep()
            # self.prob = problems.GPKick()
            # self.prob = problems.GPJump()
            self.prob = problems.GPWalk()
            self.model = model.Model(self.prob.dim, self.tasks, 'linear')
            # params = np.array([0.8647, 0.6611, -0.6017,
            #                    -0.3276, -0.3781, 0.2489])
            # params = np.array([1.0447, -0.8950, 0.0627,
            #                    -0.5505, 0.3516, 0.0807])
            # params = 0.6 * (np.random.rand(self.prob.dim * 2) - 0.5)
            # params = np.array([0.1876, -0.9591, 0.3144, -0.7733,
            #                    0, 0, 0, 0])
            # params = np.array([0.14, 0.1, -0.15, 0.05, -0.1,
            #                    0.0, 0.2, 0.0, -0.15, -0.1])
            # params = np.array([0.5862, -0.7469, 0.6050, 0.8530, 0.4105,
            #                    0.0083, 0.3875, -0.1697, -0.1931, -0.0878])
            # params = np.array([-0.7862, 0.0, -0.2, 0.65, -0.2,
            #                    0.0, 0.0, 0.0, 0.3, 0.0])
            # params = np.array([0.7727, -0.9123, 0.4292, 1.1197, -0.2495,
            #                    -0.0013, -0.0478, 0.1797, 0.01035, 0.0142125])
            # params = np.array([0.75, -0.9123, 0.4292, 0.9197, -0.2495,  # Kck
            #                    -0.0013, -0.0478, 0.1797, 0.01035, 0.0142125])
            # params = np.array([-1.0, 0.0, 0.0, 0.0, 0.0,  # Walk
            #                    1.6, 0.7, 0.3, 0.2, -0.3])
            # params = np.array([-1.1595, 0.0236, -0.6677, 0.6955, 0.1316,
            #                    -0.2170, 0.3605, -0.0384, -0.2491, -0.8954])

            params = np.zeros(8)
            params[0] = -1.0

            # # For individual approach for walking
            # # lhs = np.array([-0.57016213, 0.90000273, -0.682361, -0.333242,
            # #                 0.34159808, 0.04885952, 0.35506013])
            # lhs = np.array([-0.80708114, 0.69733204, 0.10948302, 0.13978436,
            #                 0.19853263, -0.43980252, 0.0329204])
            # rhs = np.array([-0.91217336, 0.07142092, 0.15365855, 0.01285273,
            #                 0.32689401, -0.20036001, 0.57687862])
            # params[:7] = lhs
            # params[7:] = rhs - lhs
            # # params[:7] = 0.5 * (lhs + rhs)

            self.model.mean.set_params(params)
        else:
            import json
            with open(filename) as fp:
                data = json.load(fp)
                print data
                # Parse and print
                self.prob = eval(data['prob'])
                print('Problem: %s' % self.prob)

                mean_type = eval(data['mean_type'])
                print('Mean_type: %s' % mean_type)
                self.model = model.Model(self.prob.dim, self.tasks, mean_type)

                mean_params = eval('np.' + data['mean_params'])
                print('params: %s' % mean_params)
                self.model.mean.set_params(mean_params)
        print('model:\n%s\n' % self.model)
        print('Problem name = %s' % self.prob_name())
        self.initUI()
        self.initActions()
        self.initToolbar()
        self.initMenu()

        self.idleTimer = QtCore.QTimer()
        self.idleTimer.timeout.connect(self.idleTimerEvent)
        self.idleTimer.start(0)

        self.renderTimer = QtCore.QTimer()
        self.renderTimer.timeout.connect(self.renderTimerEvent)
        self.renderTimer.start(25)

        self.cam0Event()
        self.taskSpinEvent(0.0)

        self.after_reset = True

    def initUI(self):
        self.setGeometry(0, 0, 1280, 720)
        # self.setWindowTitle('Toolbar')

        self.glwidget = GLWidget(self)
        self.glwidget.setGeometry(0, 30, 1280, 720)
        self.glwidget.prob = self.prob

    def initActions(self):
        # Create actions
        self.resetAction = QtGui.QAction('Reset', self)
        self.resetAction.triggered.connect(self.resetEvent)

        self.playAction = QtGui.QAction('Play', self)
        self.playAction.setCheckable(True)
        self.playAction.setShortcut('Space')

        self.animAction = QtGui.QAction('Anim', self)
        self.animAction.setCheckable(True)

        self.captureAction = QtGui.QAction('Capture', self)
        self.captureAction.setCheckable(True)

        self.movieAction = QtGui.QAction('Movie', self)
        self.movieAction.triggered.connect(self.movieEvent)

        self.screenshotAction = QtGui.QAction('Screenshot', self)
        self.screenshotAction.triggered.connect(self.screenshotEvent)

        self.exportAction = QtGui.QAction('Export .mtn', self)
        self.exportAction.triggered.connect(self.exportEvent)

        # Camera Menu
        self.cam0Action = QtGui.QAction('Camera0', self)
        self.cam0Action.triggered.connect(self.cam0Event)

        self.cam1Action = QtGui.QAction('Camera1', self)
        self.cam1Action.triggered.connect(self.cam1Event)

        self.printCamAction = QtGui.QAction('Print Camera', self)
        self.printCamAction.triggered.connect(self.printCamEvent)

    def initToolbar(self):
        # Create a toolbar
        self.toolbar = self.addToolBar('Control')
        self.toolbar.addAction(self.resetAction)
        self.taskSpin = QtGui.QDoubleSpinBox(self)
        self.taskSpin.setRange(0.0, 1.0)
        self.taskSpin.setDecimals(4)
        self.taskSpin.setSingleStep(0.2)
        self.taskSpin.valueChanged[float].connect(self.taskSpinEvent)
        self.toolbar.addWidget(self.taskSpin)
        self.toolbar.addAction(self.playAction)
        self.toolbar.addAction(self.animAction)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.screenshotAction)
        self.toolbar.addAction(self.captureAction)
        self.toolbar.addAction(self.movieAction)

        self.rangeSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.rangeSlider.valueChanged[int].connect(self.rangeSliderEvent)
        self.toolbar.addWidget(self.rangeSlider)

    def initMenu(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addSeparator()

        # Camera menu
        cameraMenu = menubar.addMenu('&Camera')
        cameraMenu.addAction(self.cam0Action)
        cameraMenu.addAction(self.cam1Action)
        cameraMenu.addSeparator()
        cameraMenu.addAction(self.printCamAction)

        # Recording menu
        recordingMenu = menubar.addMenu('&Recording')
        recordingMenu.addAction(self.screenshotAction)
        recordingMenu.addSeparator()
        recordingMenu.addAction(self.captureAction)
        recordingMenu.addAction(self.movieAction)
        recordingMenu.addSeparator()
        recordingMenu.addAction(self.exportAction)

    def idleTimerEvent(self):
        doCapture = False
        # Do animation
        if self.animAction.isChecked():
            v = self.rangeSlider.value() + 1
            if v <= self.rangeSlider.maximum():
                self.rangeSlider.setValue(v)
            else:
                self.animAction.setChecked(False)
            capture_rate = 20
            # if 'Walk' in repr(self.prob):
            #     capture_rate = 50
            if 'Jump' in repr(self.prob):
                capture_rate = 10
            doCapture = (v % capture_rate == 1)
        # Do play
        elif self.playAction.isChecked():
            result = self.prob.step()
            if result and self.after_reset:
                self.after_reset = False
                self.playAction.setChecked(False)

        if self.captureAction.isChecked() and doCapture:
            self.glwidget.capture(self.prob_name())

    def renderTimerEvent(self):
        self.glwidget.updateGL()
        self.statusBar().showMessage(str(self.prob))
        self.rangeSlider.setRange(0, self.prob.world.nframes - 1)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            print 'Escape key pressed! Bye.'
            self.close()

    def taskSpinEvent(self, value):
        task = self.taskSpin.value()
        self.pt = self.model.mean.point(task)
        print('task: %.4f point: %s' % (task, self.pt))
        self.prob.set_params(self.pt)

    def rangeSliderEvent(self, value):
        i = value
        if i < self.prob.world.nframes:
            self.prob.world.set_frame(i)

    def screenshotEvent(self):
        self.glwidget.capture()

    def exportEvent(self):
        self.prob.export_mtn()

    def prob_name(self):
        ret = repr(self.prob)
        ret = ret.replace('problems.', '')
        ret = ret.replace('()', '')
        task = self.taskSpin.value() if hasattr(self, 'taskSpin') else 0
        ret += '%.1f' % task
        return ret

    def movieEvent(self):
        name = self.prob_name()
        cmd = 'avconv -r 100 -i ./captures/%s.%%04d.png %s.mp4' % (name, name)
        print('cmd = %s' % cmd)
        os.system(cmd)
        # os.system('rm ./captures/*.png')

    def resetEvent(self):
        # self.prob.set_random_params()
        self.after_reset = True
        self.prob.reset()

    def cam0Event(self):
        print 'cam0Event'
        if 'Bioloid' in self.prob.skel_filename:
            if 'Walk' in self.prob_name():
                self.glwidget.tb = Trackball(phi=-25.5, theta=6.86, zoom=1,
                                             rot=[-0.11399118462862251,
                                                  -0.6947769785517393,
                                                  -0.1156975272831452,
                                                  0.7006461603991319],
                                             trans=[-0.32000000000000084,
                                                    -0.11000000000000004,
                                                    -0.710000000000003])
            elif 'Kick' in self.prob_name():
                self.glwidget.tb = Trackball(phi=-25.5, theta=6.86, zoom=1,
                                             rot=[-0.11399118462862251,
                                                  -0.6947769785517393,
                                                  -0.1156975272831452,
                                                  0.7006461603991319],
                                             trans=[-0.32000000000000084,
                                                    -0.11000000000000004,
                                                    -0.910000000000003])
            else:
                # self.glwidget.tb = Trackball(phi=-25.5, theta=6.86, zoom=1,
                #                              rot=[-0.11399118462862251,
                #                                   -0.6947769785517393,
                #                                   -0.1156975272831452,
                #                                   0.7006461603991319],
                #                              trans=[-0.06000000000000084,
                #                                     -0.17000000000000004,
                #                                     -0.710000000000003])
                self.glwidget.tb = Trackball(phi=-2.7, theta=-0.95, zoom=1,
                                             rot=[-0.022588755364916124,
                                                  -0.6948297028012123,
                                                  -0.022955860419361884,
                                                  0.7184528242749483],
                                             trans=[-0.06000000000000084,
                                                    -0.17000000000000004,
                                                    -0.6800000000000029])
        else:
            self.glwidget.tb = Trackball(phi=2.266, theta=-15.478, zoom=1,
                                         rot=[-0.09399048175876601,
                                              -0.612401798950921,
                                              -0.0675106984290682,
                                              0.7820307740607462],
                                         trans=[-0.2700000000000008,
                                                -0.900000000000000005,
                                                -3.350000000000003])

    def cam1Event(self):
        print 'cam1Event: frontview'
        if 'Bioloid' in self.prob.skel_filename:
            if 'Walk' in self.prob_name():
                self.glwidget.tb = Trackball(phi=-25.5, theta=6.86, zoom=1,
                                             rot=[-0.11399118462862251,
                                                  -0.6947769785517393,
                                                  -0.1156975272831452,
                                                  0.7006461603991319],
                                             trans=[-0.32000000000000084,
                                                    -0.11000000000000004,
                                                    -0.970000000000003])
            elif 'Kick' in self.prob_name():
                self.glwidget.tb = Trackball(phi=-25.5, theta=6.86, zoom=1,
                                             rot=[-0.11399118462862251,
                                                  -0.6947769785517393,
                                                  -0.1156975272831452,
                                                  0.7006461603991319],
                                             trans=[-0.32000000000000084,
                                                    -0.11000000000000004,
                                                    -0.970000000000003])
            else:
                self.glwidget.tb = Trackball(phi=-25.5, theta=6.86, zoom=1,
                                             rot=[-0.11399118462862251,
                                                  -0.6947769785517393,
                                                  -0.1156975272831452,
                                                  0.7006461603991319],
                                             trans=[-0.06000000000000084,
                                                    -0.17000000000000004,
                                                    -0.970000000000003])

    def printCamEvent(self):
        print 'printCamEvent'
        print '----'
        print repr(self.glwidget.tb)
        print '----'

glutInit(sys.argv)
for i, arg in enumerate(sys.argv):
    print ('Arg %d: %s' % (i, arg))
app = QtGui.QApplication(["Sample Viewer"])
# widget = WfWidget()
# widget.show()
filename = None if len(sys.argv) == 1 else sys.argv[1]
w = MyWindow(filename)
w.show()
app.exec_()
