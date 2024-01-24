import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock

p3d.loadPrcFile("config/scene.prc")

class Scene(ShowBase):
    def __init__(self):
        super().__init__(fStartDirect=True)
        self.setFrameRate(60)
        self.dt = 1/60
        # self.setCam()

        self.loadGround()

        self.taskMgr.add(self.update, "update")

    def setFrameRate(self, frameRate):
        self.setFrameRateMeter(True)
        globalClock.setMode(p3d.ClockObject.MLimited)
        globalClock.setFrameRate(frameRate)

    def setCam(self):
        self.cam.setPos(0, 1, 10)
        self.cam.setHpr(0, -90, 0)
        self.camRef = self.camera
        self.camRef.lookAt(0, 1, 0)

    def loadGround(self):
        self.ground = self.loader.loadModel("assets/scene/ground.egg")
        self.ground.reparentTo(self.render)
        self.ground.setScale(100, 1, 100)
        self.ground.setPos(0, -1, 0)
        self.ground.setTexScale(p3d.TextureStage.getDefault(), 50, 50)

    def update(self, task: p3d.PythonTask) -> int:
        return task.cont
