import panda3d.core as p3d
from direct.showbase.DirectObject import DirectObject
from direct.showbase.Loader import Loader

import numpy as np

class Humanoid(DirectObject):
    def __init__(self, loader: Loader, render: p3d.NodePath):
        self.loader = loader
        self.render = render

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
        # joint name index
        self.name2idx = {name: i for i, name in enumerate(self.jointNames)}

    def addColorTex(self, rgba: list, texName: str) -> p3d.Texture:
        # Add a single color texture
        image = p3d.PNMImage(1, 1)
        image.fill(*rgba[:3])
        image.alphaFill(rgba[3])
        tex = p3d.Texture(texName)
        tex.load(image)
        return tex

    def loadJoint(self, index: int, pos: np.ndarray, isEnd: bool) -> p3d.NodePath:
        # Load a joint with global position
        cube = self.loader.loadModel("assets/character/cube.egg")
        cube.setTextureOff(1)
        if isEnd:
            cube.setTexture(self.endTex, 1)
        cube.setScale(0.01)

        node = self.render.attachNewNode(f"joint{index}")
        cube.reparentTo(node)
        node.setPos(self.render, *pos)
        return node
    
    def loadBody(self, index: int, pos: np.ndarray, scale: np.ndarray) -> p3d.NodePath:
        # Load a body with global position and size
        cube = self.loader.loadModel("assets/character/cube.egg")
        cube.setTextureOff(1)
        cube.setTexture(self.bodyTex, 1)
        cube.setScale(*scale)

        node = self.render.attachNewNode(f"body{index}")
        cube.reparentTo(node)
        node.setPos(self.render, *pos)
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
        self.joints[self.name2idx[name]].setPos(self.render, *pos)

    def setJointRotByName(self, name, quat):
        self.joints[self.name2idx[name]].setQuat(
            self.render, p3d.Quat(*quat[..., [3, 0, 1, 2]].tolist())
        )
