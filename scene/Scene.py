import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.filter.CommonFilters import CommonFilters

p3d.loadPrcFile("config/scene.prc")

class Scene(ShowBase):
    def __init__(self):
        super().__init__(fStartDirect=True)
        self.setFrameRate(60)
        self.setCam()

        self.loadGround()
        self.loadLight()

        self.taskMgr.add(self.update, "update")

    def setFrameRate(self, frameRate):
        self.setFrameRateMeter(True)
        globalClock.setMode(p3d.ClockObject.M_limited)
        globalClock.setFrameRate(frameRate)

    def setCam(self):
        self.cam.setPos(0, 1, 10)
        self.cam.setHpr(0, -90, 0)

    def loadGround(self):
        self.ground = self.loader.loadModel("assets/scene/ground.egg")
        self.ground.reparentTo(self.render)
        self.ground.setScale(100, 1, 100)
        self.ground.setPos(0, -1, 0)
        self.ground.setTexScale(p3d.TextureStage.getDefault(), 50, 50)

    def loadLight(self):
        self.set_background_color((0, 0, 0, 1))
        self.render.setShaderAuto()
        # directional light node
        directLight = p3d.DirectionalLight("directLight")
        directLight.setColor((0.5, 0.5, 0.5, 1))
        directLight.setColorTemperature(6500)
        directLight.setShadowCaster(True, 2048, 2048)
        directLight.getLens().setFilmSize((100, 100))
        directLight.getLens().setNearFar(0, 100)
        self.directLight = self.render.attachNewNode(directLight)
        self.directLight.setPos(0, 5, 0)
        self.directLight.setHpr(0, -180, 0)
        self.render.setLight(self.directLight)
        # ambient light node
        ambientLight = p3d.AmbientLight("ambientLight")
        ambientLight.setColor((0.1, 0.1, 0.1, 1))
        self.ambientLight = self.render.attachNewNode(ambientLight)
        self.render.setLight(self.ambientLight)
        # bloom filter
        filters = CommonFilters(self.win, self.cam)
        filters.setBloom(size="small")

    def update(self, task: p3d.PythonTask) -> int:
        return task.cont
