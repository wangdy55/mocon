from direct.showbase.DirectObject import DirectObject
import numpy as np

class Character:
    def __init__(self, model: DirectObject):
        self.model = model

    def getJointPos(self):
        jointPos = [joint.getPos() for joint in self.model.getJoints()]
        jointPos = np.array(jointPos)
        return jointPos
    
    def getJointQuat(self):
        jointQuat = [joint.getQuat() for joint in self.model.getJoints()]
        jointQuat = np.array(jointQuat)
        return jointQuat

    def updateJoints(self, jointNames, jointTrans, jointOrien):
        # Update joint position and orientation in a frame
        for i in range(len(jointNames)):
            name = jointNames[i]
            self.model.setJointByName(name, jointTrans[i], jointOrien[i])