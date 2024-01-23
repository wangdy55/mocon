import panda3d.core as p3d
from direct.showbase.DirectObject import DirectObject
import math

from scene.Scene import Scene

class CameraController(DirectObject):
    def __init__(self, model: DirectObject, scene: Scene):
        super().__init__()
        self.model = model
        self.scene = scene
        self.scene.disableMouse()
        self.scene.taskMgr.add(self.update, "updateCam")
        self.cam = self.scene.cam

        self.pos = p3d.LVector3(3, 3, 3)
        self.center = p3d.LVector3(0, 1, 0)
        self.up = p3d.LVector3(0, 1, 0)
        self._lockedInfo = (
            p3d.LVector3(self.pos),
            p3d.LVector3(self.center),
            p3d.LVector3(self.up),
        )
        self._lockedMousePos = None
        self._mouseSig = 0

        self.setMouseMap()

        self.look()

    @property
    def _mousePos(self):
        return p3d.LVector2(
            self.scene.mouseWatcherNode.getMouseX(),
            self.scene.mouseWatcherNode.getMouseY()
        )

    def _lockMouseInfo(self):
        self._lockedInfo = (
            p3d.LVector3(self.pos),
            p3d.LVector3(self.center),
            p3d.LVector3(self.up),
        )
        self._lockedMousePos = self._mousePos

    def look(self):
        self.cam.setPos(self.pos)
        self.cam.lookAt(self.center, self.up)

    def setMouseMap(self):
        self.accept("mouse1", self.handleLeft, [1])
        self.accept("mouse1-up", self.handleLeft, [0])
        self.accept("mouse2", self.handleMiddle, [1])
        self.accept("mouse2-up", self.handleMiddle, [0])
        self.accept("mouse3", self.handleRight, [1])
        self.accept("mouse3-up", self.handleRight, [0])
        self.accept("wheel_down", self.handleScroll, [1])
        self.accept("wheel_up", self.handleScroll, [-1])

    def handleLeft(self, state: int):
        self._lockMouseInfo()
        if state == 1:
            self._mouseSig = 1
        else:
            self._mouseSig = 0

    def handleMiddle(self, state: int):
        self._lockMouseInfo()
        if state == 1:
            self._mouseSig = 2
        else:
            self._mouseSig = 0

    def handleRight(self, state: int):
        self._lockMouseInfo()
        if state == 1:
            self._mouseSig = 3
        else:
            self._mouseSig = 0

    def handleScroll(self, direction: int):
        # Zoom in and out
        r = self.pos - self.center
        k = 1.1 if direction > 0 else 0.9
        self.pos = self.center + r * k
        self.look()

    def update(self, task) -> int:
        if self._mouseSig == 0:
            return task.cont
        
        mousePosOff = self._mousePos - self._lockedMousePos
        if self._mouseSig == 1:
            self.pan(mousePosOff)
        elif self._mouseSig == 2:
            self.zoom(mousePosOff)
        elif self._mouseSig == 3:
            self.shift(mousePosOff)

        self.look()
        return task.cont
    
    def pan(self, mousePosOff: p3d.LVector2f):
        _pos, _center, _up = self._lockedInfo

        z = _pos - _center
        x = _up.cross(z)
        x.normalize()
        y = z.cross(x)
        y.normalize()

        z1 = _up * z.dot(_up) # vertical comp. of z
        z2 = z - z1 # horizontal comp. of z

        # initial state of rot. around x-axis, deg.
        xr0 = math.acos(z2.length() / z.length()) * (180. / math.pi)
        xr0 = -xr0 if z.dot(_up) < 0 else xr0
        # Calc. xr, x-axis rotation angle
        k = 200. # amplify factor
        xr = -mousePosOff.getY() * k + xr0
        if xr > 89: xr = 89
        if xr < -89: xr = -89
        xr -= xr0

        xRotMat = p3d.LMatrix3()
        xRotMat.setRotateMat(-xr, x, p3d.CS_yup_right)

        yRotMat = p3d.LMatrix3()
        yRotMat.setRotateMat(-mousePosOff.getX() * k, y, p3d.CS_yup_right)

        self.pos = self.center + (xRotMat * yRotMat).xform(z)

    def zoom(self, mousePosOff: p3d.LVector2f):
        _pos, _center, _ = self._lockedInfo
        z = _pos - _center
        scale = 1
        scale = 1. + scale * mousePosOff.getY()
        scale = 0.05 if scale < 0.05 else scale
        
        self.pos = _center + (z * scale)

    def shift(self, mousePosOff: p3d.LVector2f):
        _pos, _center, _up = self._lockedInfo
        z = _pos - _center
        z.normalize()
        x = _up.cross(z)
        x.normalize()
        y = z.cross(x)

        k = 0.5 * z.length() # shift scale factor
        shift = x * -mousePosOff.getX() + y * -mousePosOff.getY()
        shift *= k
        self.pos = _pos + shift
        self.center = _center + shift