import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.character.Character import Character
from mocon.controller.CharacterController import CharacterController
from scene.Scene import Scene

class MotionController:
    def __init__(
        self,
        chara: Character,
        chara_ctrl: CharacterController,
        scene: Scene,
        npz_path: str,
        mvae_path: str
    ):
        self.chara = chara
        self.chara_ctrl = chara_ctrl
        self.scene = scene
        self.frame_idx = self.chara.frame_idx
        # Initialize motion
        mvae_mocap = np.load(npz_path)
        self.num_not_ee = mvae_mocap["num_not_ee"]
        mvae_motion = mvae_mocap["mvae_motion"]
        self.motion = mvae_motion[self.frame_idx]
        
        self.load_mvae(mvae_path)

        self.scene.task_mgr.add(self.update, "update_motion_controller")

    @torch.no_grad()
    def load_mvae(self, mvae_path: str):
        self.mvae = torch.load(mvae_path)
        self.mvae.eval()

    def sync_character(self):
        vx, vz, wy = self.motion[:3]
        # local_joint_trans = self.motion[3:3+(self.num_not_ee-1)*3]
        local_joint_vec6d = self.motion[-(self.num_not_ee-1)*6:]

        # local joint orien. -> joint rotation
        local_joint_orien = np.zeros([self.chara.num_joints, 4])
        local_joint_orien[..., 3] = 1.
        local_joint_vec6d = local_joint_vec6d.reshape(self.num_not_ee-1, 3, 2)
        local_joint_z = np.cross(
            local_joint_vec6d[..., 0],
            local_joint_vec6d[..., 1],
            axis=-1
        ).reshape(self.num_not_ee-1, 3, 1)
        local_joint_z /= np.linalg.norm(local_joint_z, axis=-2, keepdims=True)
        local_joint_mat = np.concatenate(
            (local_joint_vec6d, local_joint_z),
            axis=-1
        )

        j = 0
        for i in range(1, self.chara.num_joints):
            if (self.chara.channels[i] == 0): # Skip end effectors
                continue
            local_joint_orien[i] = R.from_matrix(local_joint_mat[j]).as_quat()
            j += 1

        sync_motion = {
            "root_vel": np.array([vx, 0, vz]),
            "root_avel_y": wy,
            "local_joint_orien": local_joint_orien,
            # "joint_trans": local_joint_trans,
        }
        self.chara.sync(sync_motion)

    @torch.no_grad()
    def update_motion(self):
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

    def sync_controller(self):
        # Sync. controller to character
        self.chara_ctrl.node.setX(self.chara.root_pos[0])
        self.chara_ctrl.node.setZ(self.chara.root_pos[2])

    def update(self, task):
        self.sync_character()
        self.update_motion()
        self.sync_controller()

        return task.cont
