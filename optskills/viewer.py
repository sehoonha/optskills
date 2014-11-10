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
import problems


class MyWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        # Check and create captures directory
        if not os.path.isdir('captures'):
            os.makedirs('captures')

        # Create a simulation
        self.prob = problems.GPBow()
        self.prob.set_random_params()

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
        # self.sim.load('gp_step_1.5.plan')
        # self.sim.load('gp_step_5.plan')
        # self.sim.load('test.plan')

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

    def idleTimerEvent(self):
        doCapture = False
        # Do animation
        if self.animAction.isChecked():
            v = self.rangeSlider.value() + 1
            if v <= self.rangeSlider.maximum():
                self.rangeSlider.setValue(v)
            else:
                self.animAction.setChecked(False)
            doCapture = (v % 4 == 1)
        # Do play
        elif self.playAction.isChecked():
            result = self.prob.step()
            if result:
                self.playAction.setChecked(False)

        if self.captureAction.isChecked() and doCapture:
            self.glwidget.capture()

    def renderTimerEvent(self):
        self.glwidget.updateGL()
        self.statusBar().showMessage(str(self.prob))
        # self.rangeSlider.setRange(0, len(self.sim) - 1)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            print 'Escape key pressed! Bye.'
            self.close()

    def rangeSliderEvent(self, value):
        i = value
        if i < self.prob.world.nframes:
            self.prob.world.set_frame(i)

    def screenshotEvent(self):
        self.glwidget.capture()

    def movieEvent(self):
        os.system('avconv -r 100 -i ./captures/frame.%04d.png output.mp4')
        os.system('rm ./captures/frame.*.png')

    def resetEvent(self):
        self.prob.set_random_params()
        self.prob.reset()

    def cam0Event(self):
        print 'cam0Event'
        self.glwidget.tb = Trackball(phi=2.266, theta=-15.478, zoom=1,
                                     rot=[-0.09399048175876601,
                                          -0.612401798950921,
                                          -0.0675106984290682,
                                          0.7820307740607462],
                                     trans=[-0.5100000000000008,
                                            -0.060000000000000005,
                                            -1.320000000000003])

    def cam1Event(self):
        print 'cam1Event: frontview'
        self.glwidget.tb = Trackball(phi=3.2785, theta=-28.967, zoom=1,
                                     rot=[-0.2488669109614876,
                                          -0.04137340091750815,
                                          0.018840288369721358,
                                          0.9674701782789754],
                                     trans=[-0.290000000000001,
                                            -0.05000000000000002,
                                            -1.2100000000000029])

    def printCamEvent(self):
        print 'printCamEvent'
        print '----'
        print repr(self.glwidget.tb)
        print '----'

glutInit(sys.argv)
app = QtGui.QApplication(["Sample Viewer"])
# widget = WfWidget()
# widget.show()
w = MyWindow()
w.show()
app.exec_()
