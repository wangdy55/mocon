import panda3d.core as p3d
from direct.showbase.DirectObject import DirectObject
import math

from mocon.character.Character import Character
from scene.Scene import Scene

class CameraController(DirectObject):
    def __init__(
        self,
        chara: Character,
        scene: Scene
    ):
        super().__init__()
        self.chara = chara
        self.scene = scene
        self.scene.disable_mouse()
        self.scene.task_mgr.add(self.update, "update_cam")
        self.cam = self.scene.cam

        self.pos = p3d.LVector3(3, 3, 3)
        self.center = p3d.LVector3(0, 1, 0)
        self.up = p3d.LVector3(0, 1, 0)
        self._locked_info = (
            p3d.LVector3(self.pos),
            p3d.LVector3(self.center),
            p3d.LVector3(self.up),
        )
        self._locked_mouse_pos = None
        self._mouse_signal = 0

        self._set_mouse_map()
        self.look()

    @property
    def _mouse_pos(self):
        return p3d.LVector2(
            self.scene.mouseWatcherNode.get_mouse_x(),
            self.scene.mouseWatcherNode.get_mouse_y()
        )

    def _lock_mouse_info(self):
        self._locked_info = (
            p3d.LVector3(self.pos),
            p3d.LVector3(self.center),
            p3d.LVector3(self.up),
        )
        self._locked_mouse_pos = self._mouse_pos

    def look(self):
        self.cam.set_pos(self.pos)
        self.cam.look_at(self.center, self.up)

    @property
    def cam_fwd(self):
        return self.center - self.pos

    def _set_mouse_map(self):
        self.accept("mouse1", self.handle_left, [1])
        self.accept("mouse1-up", self.handle_left, [0])
        self.accept("mouse2", self.handle_mid, [1])
        self.accept("mouse2-up", self.handle_mid, [0])
        self.accept("mouse3", self.handle_right, [1])
        self.accept("mouse3-up", self.handle_right, [0])
        self.accept("wheel_down", self.handle_scroll, [1])
        self.accept("wheel_up", self.handle_scroll, [-1])

    def handle_left(self, state: int):
        self._lock_mouse_info()
        if state == 1:
            self._mouse_signal = 1
        else:
            self._mouse_signal = 0

    def handle_mid(self, state: int):
        self._lock_mouse_info()
        if state == 1:
            self._mouse_signal = 2
        else:
            self._mouse_signal = 0

    def handle_right(self, state: int):
        self._lock_mouse_info()
        if state == 1:
            self._mouse_signal = 3
        else:
            self._mouse_signal = 0

    def handle_scroll(self, direction: int):
        # Zoom in and out
        radius = self.pos - self.center
        k = 1.1 if direction > 0 else 0.9
        self.pos = self.center + radius * k
        self.look()

    def update(self, task) -> int:
        if self._mouse_signal == 0:
            return task.cont
        
        mouse_pos_delta = self._mouse_pos - self._locked_mouse_pos
        if self._mouse_signal == 1:
            self.pan(mouse_pos_delta)
        elif self._mouse_signal == 2:
            self.zoom(mouse_pos_delta)
        elif self._mouse_signal == 3:
            self.shift(mouse_pos_delta)

        self.look()
        return task.cont
    
    def pan(self, mouse_pos_delta: p3d.LVector2f):
        _pos, _center, _up = self._locked_info

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
        xr = -mouse_pos_delta.getY() * k + xr0
        if xr > 89: xr = 89
        if xr < -89: xr = -89
        xr -= xr0

        x_rot_mat = p3d.LMatrix3()
        x_rot_mat.setRotateMat(
            -xr, x, p3d.CS_yup_right
        )

        y_rot_mat = p3d.LMatrix3()
        y_rot_mat.setRotateMat(
            -mouse_pos_delta.get_x() * k,
            y,
            p3d.CS_yup_right
        )

        self.pos = self.center + (x_rot_mat * y_rot_mat).xform(z)

    def zoom(self, mouse_pos_delta: p3d.LVector2f):
        pos0, center0, _ = self._locked_info
        z = pos0 - center0
        scale = 1
        scale = 1. + scale * mouse_pos_delta.get_y()
        scale = 0.05 if scale < 0.05 else scale
        
        self.pos = center0 + (z * scale)

    def shift(self, mouse_pos_delta: p3d.LVector2f):
        pos0, center0, _up = self._locked_info
        z = pos0 - center0
        z.normalize()
        x = _up.cross(z)
        x.normalize()
        y = z.cross(x)

        k = 0.5 * z.length() # shift scale factor
        shift = x * -mouse_pos_delta.get_x() + y * -mouse_pos_delta.get_y()
        shift *= k
        self.pos = pos0 + shift
        self.center = center0 + shift
        