import panda3d.core as p3d
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock

from scene.Model import Model

p3d.loadPrcFile("config/scene.prc")

class Scene(ShowBase):
    def __init__(self):
        super().__init__(fStartDirect=True)
        self.setFrameRate(60)
        self.setCam()

        self.loadGround()
        self.loadModel()
        self.loadLight()

        self.taskMgr.add(self.update, "update")

    def setFrameRate(self, frameRate):
        self.setFrameRateMeter(True)
        globalClock.setMode(p3d.ClockObject.MLimited)
        globalClock.setFrameRate(frameRate)

    def setCam(self):
        # self.cam.setPos(0, 1, 10)
        # self.cam.setHpr(0, -90, 0)
        self.camRef = self.camera
        # self.camRef.lookAt(0, 1, 0)

    def loadGround(self):
        self.ground = self.loader.loadModel("assets/scene/ground.egg")
        self.ground.reparentTo(self.render)
        self.ground.setScale(100, 1, 100)
        self.ground.setPos(0, -1, 0)
        self.ground.setTexScale(p3d.TextureStage.getDefault(), 50, 50)

    def loadModel(self):
        self.model = Model(self.loader, self.render)

    def loadLight(self):
        self.set_background_color((0, 0, 0, 1))
        self.render.setShaderAuto(True)

        ambientLight = p3d.AmbientLight('ambientLight')
        ambientLight.setColor((0.3, 0.3, 0.3, 1))
        self.alnp = self.render.attachNewNode(ambientLight)
        self.render.setLight(self.alnp)
        
        # directional light set
        self.directLightSet = p3d.NodePath("directLightSet")

        fillLight1 = p3d.DirectionalLight('fillLight1')
        fillLight1.setColor((0.45, 0.45, 0.45, 1))
        flnp1 = self.directLightSet.attachNewNode(fillLight1)
        flnp1.setPos(10, 10, 10)
        flnp1.lookAt((0, 0, 0), (0, 1, 0))
        self.render.setLight(flnp1)
        
        fillLight2 = p3d.DirectionalLight("fillLight2")
        fillLight2.setColor((0.45, 0.45, 0.45, 1))
        flnp2 = self.directLightSet.attachNewNode(fillLight2)
        flnp2.setPos(-10, 10, 10)
        flnp2.lookAt((0, 0, 0), (0, 1, 0))
        self.render.setLight(flnp2)

        keyLight = p3d.DirectionalLight("keyLight")
        keyLight.setColorTemperature(6500)
        keyLight.setShadowCaster(True, 2048, 2048)
        keyLight.getLens().setFilmSize((10, 10))
        keyLight.getLens().setNearFar(0.1, 300)
        klnp = self.directLightSet.attachNewNode(keyLight)
        klnp.setPos(0, 20, -10)
        klnp.lookAt((0, 0, 0), (0, 1, 0))
        self.render.setLight(klnp)

        self.directLightSet.wrtReparentTo(self.model.root)

    def update(self, task: p3d.PythonTask) -> int:
        return task.cont
