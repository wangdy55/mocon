from direct.showbase.DirectObject import DirectObject
import numpy as np
from scipy.spatial.transform import Rotation as R

from scene.Scene import Scene
from mocon.motion.BVHMotion import BVHMotion
from mocon.motion.utils.QuatHelper import QuatHelper

class Character:
    def __init__(self, model: DirectObject, scene: Scene, bvhPath: str):
        self.model = model
        self.scene = scene

        self.bvhMotion = BVHMotion(bvhPath)
        self.loadBVHInfo(startFrame=300)

        self.scene.taskMgr.add(self.update, "updateCharacter")

    def loadBVHInfo(self, startFrame: int):
        self.dt = self.bvhMotion.dt
        self.numJoints = self.bvhMotion.numJoints
        self.parentIdx = self.bvhMotion.parentIdx
        self.offsets = self.bvhMotion.offsets
        self.channels = self.bvhMotion.channels
        self.startFrame = startFrame

        # Initialize joint trans. and orien. at a start frame
        jointPos = self.bvhMotion.jointPos[startFrame][None, ...]
        jointPos[:, 0, [0, 2]] = 0. # Move the root joint to over the origin
        jointQuat = self.bvhMotion.jointQuat[startFrame][None, ...]
        jointTrans, jointOrien = self.bvhMotion.batchForwardKinematics(
            jointPos, jointQuat
        )
        self.jointTrans = jointTrans.squeeze()
        self.jointOrien = jointOrien.squeeze()

    def update(self, task):
        self.updateModel(
            self.bvhMotion.jointNames, self.jointTrans, self.jointOrien
        )
        return task.cont

    def updateState(self, fMotion: dict):
        # Update character state with motion
        rootVel = fMotion["rootVel"]
        rootAvelY = fMotion["rootAvelY"]
        localJointOrien = fMotion["localJointOrien"]

        rootPos = self.jointTrans[0]
        rootQuat = self.jointOrien[0]
        rootRot = R.from_quat(rootQuat)

        rootPos = rootPos + rootRot.apply(rootVel) * self.dt
        rootAngleY = QuatHelper.yExtractRad(rootQuat[None, ...])
        rootAngleY = rootAngleY.squeeze() + rootAvelY * self.dt
        rootQuat = R.from_euler("Y", rootAngleY, degrees=False).as_quat()

        # jointPos & jointRot
        jointPos = np.zeros((self.numJoints, 3))
        jointPos[1:] = self.offsets[1:]
        jointPos[0] = rootPos

        jointQuat = np.zeros_like(localJointOrien)
        jointQuat[..., 3] = 1.
        for i, p in enumerate(self.parentIdx):
            if p == -1: continue
            Rp = R.from_quat(localJointOrien[p])
            Ri = Rp.inv() * R.from_quat(localJointOrien[i])
            jointQuat[i] = Ri.as_quat()
        jointQuat[0, :] = rootQuat

        # Forward kinematics
        jointTrans, jointOrien = self.bvhMotion.batchForwardKinematics(
            jointPos[None, ...], jointQuat[None, ...]
        )
        self.jointTrans = jointTrans.squeeze()
        self.jointOrien = jointOrien.squeeze()

    def updateModel(self, jointNames, jointTrans, jointOrien):
        # Update joint position and orientation in a frame
        for i in range(len(jointNames)):
            name = jointNames[i]
            self.model.setJointByName(name, jointTrans[i], jointOrien[i])

    def _getJointPos(self):
        jointPos = [joint.getPos() for joint in self.model.getJoints()]
        jointPos = np.array(jointPos)
        return jointPos
    
    def _getJointQuat(self):
        jointQuat = [joint.getQuat() for joint in self.model.getJoints()]
        jointQuat = np.array(jointQuat)
        return jointQuat