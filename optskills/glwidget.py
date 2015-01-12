from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtOpenGL import *
import pydart
import gltools


class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.width = 1280
        self.height = 720

        self.tb = pydart.Trackball()
        self.lastPos = None
        self.zoom = -1.2

        self.prob = None
        self.captureIndex = 0

    def sizeHint(self):
        return QtCore.QSize(self.width, self.height)

    def paintGL(self):
        glEnable(GL_DEPTH_TEST)
        # glClearColor(0.95, 0.95, 0.95, 0.0)
        glClearColor(0.90, 0.90, 0.90, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        # glTranslate(0.0, -0.2, self.zoom)  # Camera
        glTranslate(*self.tb.trans)
        glMultMatrixf(self.tb.matrix)

        glPushMatrix()
        # gltools.render_axis(10)
        # Draw chess board
        gltools.glMove([0.0, -0.01, 0.0])
        gltools.render_chessboard(10, 20.0)

        # gltools.glMove([0, 0, 0])
        # self.prob.world.skels[1].render()

        for i, skel in enumerate(self.prob.world.skels):
            if i == 0:
                continue
            # Draw skeleton
            gltools.glMove([0, 0, 0])
            glPushMatrix()
            M_s = [1.0, 0.0, 0.0, 0.0,
                   1.0, 0.0, -1.0, 0.0,
                   0.0, 0.0, 1.0, 0.0,
                   0.0, -0.001, 0.0, 1.0]
            glMultMatrixf(M_s)
            skel.render_with_color(0.0, 0.0, 0.0)
            glPopMatrix()
            skel.render()
        # self.prob.render()
        glPopMatrix()

    def resizeGL(self, w, h):
        (self.width, self.height) = (w, h)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(45.0, float(w) / float(h), 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def initializeGL(self):
        # glDisable(GL_CULL_FACE)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)

        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        # glEnable(GL_POLYGON_SMOOTH)
        # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_DITHER)
        glShadeModel(GL_SMOOTH)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        # glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 20.0, -20.0, 0.0])
        # glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.09, 0.09, 0.09, 1.0])
        # glLightfv(GL_LIGHT1, GL_POSITION, [10.0, 10.0, 10.0, 0.0])
        # glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.09, 0.09, 0.09, 1.0])
        # glEnable(GL_LIGHT0)
        # glEnable(GL_LIGHT1)
        # glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 0.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [-1.0, -1.0, 0.0, 0.0])
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHTING)

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        self.lastPos = None

    def mouseMoveEvent(self, event):
        # (w, h) = (self.width, self.height)
        x = event.x()
        y = event.y()
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        modifiers = QtGui.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            self.tb.zoom_to(dx, -dy)
        elif modifiers == QtCore.Qt.ControlModifier:
            self.tb.trans_to(dx, -dy)
        else:
            self.tb.drag_to(x, y, dx, -dy)
        self.lastPos = event.pos()
        self.updateGL()

    def capture(self, name='frame'):
        img = self.grabFrameBuffer()
        filename = 'captures/%s.%04d.png' % (name, self.captureIndex)
        img.save(filename)
        print 'Capture to ', filename
        self.captureIndex += 1
