import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.character.Character import Character
from mocon.controller.CharacterController import CharacterController
from mocon.motion.mvae.model.MotionVAE import MotionMixtureSpecialistVAE
from scene.Scene import Scene

class MotionController:
    def __init__(
        self,
        character: Character,
        characterController: CharacterController,
        scene: Scene,
        npzPath: str,
        mvaePath: str
    ):
        self.character = character
        self.characterController = characterController
        self.scene = scene
        self.startFrame = self.character.startFrame
        # Initialize motion
        mvaeMocap = np.load(npzPath)
        self.numNotEe = mvaeMocap["numNotEe"]
        mvaeMotion = mvaeMocap["mvaeMotion"]
        self.motion = mvaeMotion[self.startFrame]
        
        self.loadMVAE(mvaePath)

        self.scene.taskMgr.add(self.update, "updateMotionController")

    @torch.no_grad()
    def loadMVAE(self, mvaePath: str):
        self.mvae = torch.load(mvaePath)
        self.mvae.eval()

    def updateCharacter(self):
        # Call Character to update state
        vx, vz, wy = self.motion[:3]
        # localJointTrans = self.motion[3:3+(self.numNotEe-1)*3]
        localJointVec6d = self.motion[-(self.numNotEe-1)*6:]

        # local joint orien. -> joint rotation
        localJointOrien = np.zeros([self.character.numJoints, 4])
        localJointOrien[..., 3] = 1.
        localJointVec6d = localJointVec6d.reshape(self.numNotEe-1, 3, 2)
        localJointZ = np.cross(
            localJointVec6d[..., 0], localJointVec6d[..., 1], axis=-1
        ).reshape(self.numNotEe-1, 3, 1)
        localJointZ /= np.linalg.norm(localJointZ, axis=-2, keepdims=True)
        localJointMat = np.concatenate((localJointVec6d, localJointZ), axis=-1)

        j = 0
        for i in range(1, self.character.numJoints):
            if (self.character.channels[i] == 0): # Skip end effectors
                continue
            localJointOrien[i] = R.from_matrix(localJointMat[j]).as_quat()
            j += 1

        fMotion = { # format motion for Character
            "rootVel": np.array([vx, 0, vz]),
            "rootAvelY": wy,
            "localJointOrien": localJointOrien,
            # "jointTrans": localJointTrans
        }
        self.character.updateState(fMotion)

    @torch.no_grad()
    def updateMotion(self):
        # Call MVAE to predict motion
        z = torch.normal(
            mean=0, std=1,
            size=(1, self.mvae.latent_size)
        ).to("cuda:0")
        c = torch.from_numpy(
            self.motion
        ).float().reshape(1, -1).to("cuda:0")

        c = self.mvae.normalize(c)
        output = self.mvae.sample(
            z, c, deterministic=True
        )
        output = self.mvae.denormalize(output).cpu()

        output = output.detach().numpy()
        self.motion = output.squeeze()

    def update(self, task):
        self.updateCharacter()
        self.updateMotion()

        return task.cont
