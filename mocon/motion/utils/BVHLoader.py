import os
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.QuatHelper import QuatHelper

"""
N = numFrames
M = numJoints
"""

class BVHLoader():
    def __init__(self, bvhPath: str):
        self.bvhPath = bvhPath

        # static data
        self.jointNames = []
        self.parentIdx = []
        self.channels = []
        self.offsets = []
        self.dt = 0.01

        # motion data
        self.startFrame = 0
        self.numFrames = 0
        self.frameSize = 0
        self.jointPos = None  # shape = (N, M, 3)
        self.jointRot = None  # shape = (N, M, 4), R.quat

        if self.bvhPath is not None:
            self.loadBVH()
        pass

    @property
    def numJoints(self):
        return len(self.jointNames)

    def loadStaticData(self) -> tuple:
        with open(self.bvhPath, "r") as f:
            channels = []
            jointNames = []
            parentIdx = []
            offsets = []
            endSites = []
            dt = 0

            parentStack = [None]
            for line in f:
                if "ROOT" in line or "JOINT" in line:
                    jointNames.append(line.split()[-1])
                    parentIdx.append(parentStack[-1])
                    channels.append("")
                    offsets.append([0, 0, 0])

                elif "End Site" in line:
                    endSites.append(len(jointNames))
                    jointNames.append(parentStack[-1] + "_end")
                    parentIdx.append(parentStack[-1])
                    channels.append("")
                    offsets.append([0, 0, 0])

                elif "{" in line:
                    parentStack.append(jointNames[-1])

                elif "}" in line:
                    parentStack.pop()

                elif "OFFSET" in line:
                    offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1, 3)

                elif "CHANNELS" in line:
                    transOrder = []
                    rotOrder = []
                    for token in line.split():
                        if "position" in token:
                            transOrder.append(token[0])

                        if "rotation" in token:
                            rotOrder.append(token[0])

                    channels[-1] = "".join(transOrder) + "".join(rotOrder)

                elif "Frame Time:" in line:
                    dt = float(line.split()[-1])
                    break

        parentIdx = [-1] + [jointNames.index(i) for i in parentIdx[1:]]
        channels = [len(i) for i in channels]
        offsets = np.concatenate(offsets, axis=0)
        return jointNames, parentIdx, channels, offsets, dt

    def loadMotionData(self, startFrame=0) -> np.ndarray:
        """
        @Parameters:
            startFrame: int, which frame to start from
        
        @Returns:
            motionData: ndarray, shape = (N, frame_size)
        """
        with open(self.bvhPath, "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].startswith("Frame Time"):
                    break
            motionData = []
            for line in lines[i+1:]:
                data = [float(x) for x in line.split()]
                if len(data) == 0:
                    break
                motionData.append(np.array(data).reshape(1, -1))
            motionData = np.concatenate(motionData, axis=0)
        self.startFrame = startFrame
        return motionData[startFrame:]

    def loadBVH(self):
        self.jointNames, self.parentIdx, self.channels, self.offsets, self.dt = self.loadStaticData()
        motion_data = self.loadMotionData()

        self.numFrames = motion_data.shape[0]
        self.frameSize = motion_data.shape[1]

        self.jointPos = np.zeros([self.numFrames, self.numJoints, 3])
        self.jointRot = np.zeros([self.numFrames, self.numJoints, 4])
        self.jointRot[:, :, 3] = 1.0

        channelIdx = 0
        for i in range(len(self.jointNames)):
            if self.channels[i] == 0:
                self.jointPos[:, i] = self.offsets[i].reshape(1, 3)
                continue

            if self.channels[i] == 3:
                position = self.offsets[i].reshape(1, 3)
                rotation = motion_data[:, channelIdx:channelIdx+3]
            elif self.channels[i] == 6:
                position = motion_data[:, channelIdx:channelIdx+3]
                rotation = motion_data[:, channelIdx+3:channelIdx+6]
            self.jointPos[:, i, :] = position
            self.jointRot[:, i, :] = R.from_euler("XYZ", rotation, degrees=True).as_quat()

            channelIdx += self.channels[i]

        return

    def batchForwardKinematics(self, jointPos=None, jointRot=None) -> tuple:
        """
        @Parameters:
            jointPos: ndarray, shape = (N, M, 3)
            jointRot: ndarray, shape = (N, M, 4), R.quat
        
        @Returns:
            jointTrans: ndarray, shape = (N, M, 3)
            jointOrien: ndarray, shape = (N, M, 4), R.quat
        """
        if jointPos is None:
            jointPos = self.jointPos
        if jointRot is None:
            jointRot = self.jointRot

        jointTrans = np.zeros_like(jointPos)
        jointOrien = np.zeros_like(jointRot)
        jointOrien[..., 3] = 1.0

        for i, p in enumerate(self.parentIdx):
            if p == -1:
                jointTrans[:, i] = jointPos[:, i]
                jointOrien[:, i] = jointRot[:, i]
            else:
                Op = R.from_quat(jointOrien[:, p])
                Oi = Op * R.from_quat(jointRot[:, i])
                jointTrans[:, i] = jointTrans[:, p] + Op.apply(jointPos[:, i])
                jointOrien[:, i] = Oi.as_quat()

        return jointTrans, jointOrien
   
    def adjustJointNames(self, targetJointNames):
        """
        Adjust joint seq. to target joint names
        """
        idx = [self.jointNames.index(name) for name in targetJointNames]
        idxInv = [targetJointNames.index(name) for name in self.jointNames]
        self.jointNames = [self.jointNames[i] for i in idx]
        self.parentIdx = [idxInv[self.parentIdx[i]] for i in idx]
        self.parentIdx[0] = -1
        self.joint_channel = [self.channels[i] for i in idx]
        self.jointPos = self.jointPos[:,idx,:]
        self.jointRot = self.jointRot[:,idx,:]
    
    def rawCopy(self):
        return copy.deepcopy(self)
    
    def subSeq(self, start, end):
        """
        Get a sub sequence info. of joint position and rotation

        @Parameters:
            start, end: int, frame index

        @Returns:
            res: sub sequence
        """
        res = self.rawCopy()
        res.jointPos = res.jointPos[start:end,:,:]
        res.jointRot = res.jointRot[start:end,:,:]
        return res
    
    def append(self, other):
        """
        Append a motion to the end
        """
        other = other.rawCopy()
        other.adjustJointNames(self.jointNames)
        self.jointPos = np.concatenate((self.jointPos, other.jointPos), axis=0)
        self.jointRot = np.concatenate((self.jointRot, other.jointRot), axis=0)
        pass
    
    # for MotionVAE motion format
    def getLocalJointInfo(self) -> tuple:
        """
        Get local (root based) joint info. in MotionVAE format

        @Return:
            local joint info: tuple, (Jp, Jv, Jr)
        """
        # local batch FK -> local joint position (Jp)
        localJointPos = self.jointPos.copy()
        localJointPos[:, 0] = [0., 0., 0.]
        localJointRot = self.jointRot.copy()
        localJointRot[:, 0] = [0., 0., 0., 1.]
        localJointTrans, localJointOrien = self.batchForwardKinematics(
            jointPos=localJointPos, jointRot=localJointRot
        )

        # Remove end effector info.
        notEe = [i for i, ch in enumerate(self.channels) if ch != 0]
        localJointTrans = localJointTrans[:, notEe]
        localJointOrien = localJointOrien[:, notEe]

        # Jp -> Jv
        localJointVel = (localJointTrans[1:] - localJointTrans[:-1]) / self.dt # shape = (N-1, M, 3)

        # local orientation -> local joint vec6d (Jr)
        # vec6d: first 2 col (xy) of local orientation matrix
        localJointMat = np.zeros([self.numFrames, len(notEe), 3, 3])
        for i in range(len(notEe)):
            localJointMat[:, i] = R.from_quat(localJointOrien[:, i]).as_matrix()
        localJointVec6d = localJointMat[..., :2]
        localJointVec6d = localJointVec6d.reshape(self.numFrames, len(notEe), -1)

        return localJointTrans[:-1], localJointVel, localJointVec6d[:-1]

    def saveMvaeMocap(self, npzPath: str):
        """
        Convert raw motion data to
        MVAE motion format:
            (r"x, r"z, w_y, Jp, Jv, Jr)
            Jx = (jx_1, jx_2, ..., jx_m), x = {p, v, r}
        and save as npz file

        @Parameters:
            mvae_motion_file_path: str, path to MVAE motion npz file

        @Returns:
        """
        if os.path.exists(npzPath):
            print(f"{npzPath} already exists, skip saving")
            return

        # root info: (r'x, r'z, w_y)
        rootQuat = self.jointRot[:, 0].copy()
        rootPos = self.jointPos[:, 0].copy()

        # root orientation -> w_y (rad/s)
        rootAngleY = QuatHelper.yExtractRad(rootQuat)
        rootAvelY = rootAngleY[1:] - rootAngleY[:-1]
        # Fix rotation around ±π
        rootAvelY[rootAvelY >= np.pi] -= 2 * np.pi
        rootAvelY[rootAvelY <= -np.pi] += 2 * np.pi
        rootAvelY /= self.dt

        # root translation -> r'x, r'z
        rootVel = (rootPos[1:, :3] - rootPos[:-1, :3]) / self.dt # shape = (N-1, 3)
        # global velocity -> local velocity
        rootRot = R.from_quat(rootQuat)[:-1]
        rootVelXz = rootRot.apply(rootVel, inverse=True)[:, [0, 2]]
        
        # local batch FK -> Jp, Jv, Jr
        localJointTrans, localJointVel, localJointVec6d = self.getLocalJointInfo()
        numNotEe = localJointTrans.shape[1]

        # Remove root info and reshape
        localJointTrans = localJointTrans[:, 1:].reshape(self.numFrames-1, -1)
        localJointVel = localJointVel[:, 1:].reshape(self.numFrames-1, -1)
        localJointVec6d = localJointVec6d[:, 1:].reshape(self.numFrames-1, -1)

        # Cat. -> (v_x, v_z, w_y(1), Jp(19*3), Jv(19*3), Jr(19*6)), shape = (N-1, frame_size=3+228)
        mvaeMotion = np.concatenate(
            (rootVelXz, rootAvelY.reshape(-1, 1), localJointTrans, localJointVel, localJointVec6d),
            axis=1
        )

        np.savez(npzPath,
            numFrames=self.numFrames,
            numJoints=self.numJoints,
            numNotEe=numNotEe,
            rootAngleYStart=rootAngleY[0],
            rootPosStart=rootPos[0],
            mvaeMotion=mvaeMotion
        )
        print("MVAE mocap saved to", npzPath)

    def loadMvaeMocap(self, npzPath:str) -> tuple:
        """
        Recover BVH mocap from MVAE mocap

        @Parameters:
            npzPath: str, path to local_motion.npz

        @Returns:
            jointPos: ndarray, shape = (N, M, 3)
            jointQuat: ndarray, shape = (N, M, 4), R.quat
        """

        # mvae motion -> joint position, joint rotation, for batch FK
        mvaeMocap = np.load(npzPath)
        numFrames = mvaeMocap["numFrames"]
        numJoints = mvaeMocap["numJoints"]
        numNotEe = mvaeMocap["numNotEe"]
        rootAngleYStart = mvaeMocap["rootAngleYStart"]
        rootPosStart = mvaeMocap["rootPosStart"]
        mvaeMotion = mvaeMocap["mvaeMotion"]
        # MVAE motion sliced
        rootVelXz = mvaeMotion[:, :2]
        rootAvelY = mvaeMotion[:, 2]
        localJointVec6d = mvaeMotion[:, -(numNotEe-1)*6:]

        # root rotation
        rootAngleY = np.zeros([numFrames])
        rootAngleY[0] = rootAngleYStart
        for i in range(1, numFrames):
            rootAngleY[i] = rootAngleY[i-1] + rootAvelY[i-1] * self.dt
        rootAngleY = rootAngleY.reshape(numFrames, 1)
        zeroAngle = np.zeros([numFrames, 1])
        rootEuler = np.concatenate(
            (zeroAngle, rootAngleY, zeroAngle),
            axis=1
        )
        rootEuler = rootEuler[:-1, :]
        rootRot = R.from_euler("XYZ", rootEuler, degrees=False)

        rootQuat = rootRot.as_quat()

        # root position
        zeroVel = np.zeros([numFrames-1, 1])
        rootVelX = rootVelXz[:, 0].reshape(-1, 1)
        rootVelZ = rootVelXz[:, 1].reshape(-1, 1)
        rootVel = np.concatenate((rootVelX, zeroVel, rootVelZ), axis=1)

        rootPos = np.zeros([numFrames-1, 3])
        rootPos[0] = rootPosStart
        for i in range(1, numFrames-1):
            rootPos[i] = rootPos[i-1] + rootRot[i-1].apply(rootVel[i-1]) * self.dt

        # local joint orien. -> joint rotation
        jointOrien = np.zeros([numFrames-1, numJoints, 4])
        jointOrien[..., 3] = 1.
        localJointVec6d = localJointVec6d.reshape(numFrames-1, numNotEe-1, 3, 2)
        localJointZ = np.cross(
            localJointVec6d[..., 0], localJointVec6d[..., 1], axis=-1
        ).reshape(numFrames-1, numNotEe-1, 3, 1)
        localJointZ /= np.linalg.norm(localJointZ, axis=-2, keepdims=True)
        localJointMat = np.concatenate((localJointVec6d, localJointZ), axis=-1)

        j = 0
        for i in range(1, numJoints):
            if (self.channels[i] == 0): # Skip end effectors
                continue
            jointOrien[:, i] = R.from_matrix(localJointMat[:, j]).as_quat()
            j += 1

        # jointPos & jointRot
        jointPos = np.zeros((numFrames-1, numJoints, 3))
        jointPos[:, 1:, :] = np.tile(self.offsets[1:], (self.numFrames-1, 1, 1))
        jointPos[:, 0] = rootPos

        jointQuat = np.zeros_like(jointOrien)
        jointQuat[..., 3] = 1.
        for i, p in enumerate(self.parentIdx):
            if p == -1: continue
            Rp = R.from_quat(jointOrien[:, p])
            Ri = Rp.inv() * R.from_quat(jointOrien[:, i])
            jointQuat[:, i] = Ri.as_quat()
        jointQuat[:, 0, :] = rootQuat

        return jointPos, jointQuat
