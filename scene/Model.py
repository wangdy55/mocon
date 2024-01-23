import panda3d.core as p3d
from direct.showbase.DirectObject import DirectObject
from direct.showbase.Loader import Loader
import numpy as np

from scene.Scene import Scene

class Model(DirectObject):
    def __init__(self, scene: Scene):
        self.scene = scene

        # Add texture
        endColor = [0, 1, 0, 1]
        bodyColor = [141/255, 141/255, 170/255, 1]
        self.endTex = self.addColorTex(endColor, "endTex")
        self.bodyTex = self.addColorTex(bodyColor, "bodyTex")
        # Load t-pose info.
        tPose = np.load("assets/character/humanoid.npz")
        self.jointNames = tPose["jointNames"]
        self.parentIdx = tPose["parentIdx"]
        # Load joints and bodies
        jointPos = tPose["jointPos"]
        bodyPos = tPose["bodyPos"]
        bodyScale = tPose["bodyScale"]
        self.joints, self.bodies = self.loadModel(jointPos, bodyPos, bodyScale)
        # Define the first joint as root
        self.root = self.joints[0]
        # joint name index
        self.name2idx = {name: i for i, name in enumerate(self.jointNames)}

        self.loadLight()

    def addColorTex(self, rgba: list, texName: str) -> p3d.Texture:
        # Add a single color texture
        image = p3d.PNMImage(1, 1)
        image.fill(*rgba[:3])
        image.alphaFill(rgba[3])
        tex = p3d.Texture(texName)
        tex.load(image)
        return tex
    
    def loadLight(self):
        self.scene.set_background_color((0, 0, 0, 1))
        self.scene.render.setShaderAuto(True)

        ambientLight = p3d.AmbientLight('ambientLight')
        ambientLight.setColor((0.3, 0.3, 0.3, 1))
        self.alnp = self.scene.render.attachNewNode(ambientLight)
        self.scene.render.setLight(self.alnp)
        
        # directional light set
        self.directLightSet = p3d.NodePath("directLightSet")

        fillLight1 = p3d.DirectionalLight('fillLight1')
        fillLight1.setColor((0.45, 0.45, 0.45, 1))
        flnp1 = self.directLightSet.attachNewNode(fillLight1)
        flnp1.setPos(10, 10, 10)
        flnp1.lookAt((0, 0, 0), (0, 1, 0))
        self.scene.render.setLight(flnp1)
        
        fillLight2 = p3d.DirectionalLight("fillLight2")
        fillLight2.setColor((0.45, 0.45, 0.45, 1))
        flnp2 = self.directLightSet.attachNewNode(fillLight2)
        flnp2.setPos(-10, 10, 10)
        flnp2.lookAt((0, 0, 0), (0, 1, 0))
        self.scene.render.setLight(flnp2)

        keyLight = p3d.DirectionalLight("keyLight")
        keyLight.setColorTemperature(6500)
        keyLight.setShadowCaster(True, 2048, 2048)
        keyLight.getLens().setFilmSize((10, 10))
        keyLight.getLens().setNearFar(0.1, 300)
        klnp = self.directLightSet.attachNewNode(keyLight)
        klnp.setPos(0, 20, -10)
        klnp.lookAt((0, 0, 0), (0, 1, 0))
        self.scene.render.setLight(klnp)

        self.directLightSet.wrtReparentTo(self.root)

    def loadJoint(self, index: int, pos: np.ndarray, isEnd: bool) -> p3d.NodePath:
        # Load a joint with global position
        cube = self.scene.loader.loadModel("assets/character/cube.egg")
        cube.setTextureOff(1)
        if isEnd:
            cube.setTexture(self.endTex, 1)
        cube.setScale(0.01)

        node = self.scene.render.attachNewNode(f"joint{index}")
        cube.reparentTo(node)
        node.setPos(self.scene.render, *pos)
        return node
    
    def loadBody(self, index: int, pos: np.ndarray, scale: np.ndarray) -> p3d.NodePath:
        # Load a body with global position and size
        cube = self.scene.loader.loadModel("assets/character/cube.egg")
        cube.setTextureOff(1)
        cube.setTexture(self.bodyTex, 1)
        cube.setScale(*scale)

        node = self.scene.render.attachNewNode(f"body{index}")
        cube.reparentTo(node)
        node.setPos(self.scene.render, *pos)
        return node
    
    def loadModel(self, jointPos: np.ndarray, bodyPos: np.ndarray, bodyScale: np.ndarray) -> tuple:
        # Load the model with joints and bodies
        joints, bodies = [], []
        numJoints = len(jointPos)
        numBodies = len(bodyPos)
        isEnd = ["end" in name for name in self.jointNames]

        for i in range(numJoints):
            joints.append(self.loadJoint(i, jointPos[i], isEnd[i]))
            if i >= numBodies:
                continue
            bodies.append(self.loadBody(i, bodyPos[i], bodyScale[i]))
            bodies[-1].wrtReparentTo(joints[i])

        joints = np.array(joints)
        bodies = np.array(bodies)
        return joints, bodies
    
    def getJoints(self) -> np.ndarray:
        # dtype of joints is panda3d.core.NodePath
        return self.joints
    
    def setJointPosByName(self, name, pos):
        self.joints[self.name2idx[name]].setPos(self.scene.render, *pos)

    def setJointRotByName(self, name, quat):
        self.joints[self.name2idx[name]].setQuat(
            self.scene.render, p3d.Quat(*quat[..., [3, 0, 1, 2]].tolist())
        )
