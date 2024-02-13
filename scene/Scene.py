import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock

p3d.loadPrcFile("config/scene.prc")

class Scene(ShowBase):
    def __init__(self):
        super().__init__(fStartDirect=True)
        self.set_frame_rate(100)
        self.dt = 1/100

        self.load_ground()

        self.task_mgr.add(self.update, "update")

    def set_frame_rate(self, frame_rate):
        self.set_frame_rate_meter(True)
        globalClock.set_mode(p3d.ClockObject.MLimited)
        globalClock.set_frame_rate(frame_rate)

    def set_cam(self):
        self.cam.set_pos(0, 1, 10)
        self.cam.setHpr(0, -90, 0)
        self.camRef = self.camera
        self.camRef.lookAt(0, 1, 0)

    def load_ground(self):
        self.ground = self.loader.loadModel("assets/scene/ground.egg")
        self.ground.reparent_to(self.render)
        self.ground.set_scale(100, 1, 100)
        self.ground.set_pos(0, -1, 0)
        self.ground.set_tex_scale(p3d.TextureStage.get_default(), 50, 50)

    def update(self, task: p3d.PythonTask) -> int:
        return task.cont
