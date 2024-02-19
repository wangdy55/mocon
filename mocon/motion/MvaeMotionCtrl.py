import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.character.Character import Character
from mocon.utils.QuatUtil import QuatUtil

class MvaeMotionCtrl:
    def __init__(
        self,
        chara: Character,
    ):
        self.chara = chara
        self.scene = self.chara.scene
        self.frame_idx = self.chara.frame
        # Initialize motion
        mvae_mocap = np.load(self.chara.npz_file)
        self.num_not_ee = mvae_mocap["num_not_ee"]
        mvae_motion = mvae_mocap["mvae_motion"]
        self.motion = mvae_motion[self.frame_idx]
        
        self.load_mvae(self.chara.mvae_path)
        self.z = torch.normal(
            mean=0, std=1,
            size=(1, self.mvae.latent_size)
        ).to("cuda:0")

        self.scene.task_mgr.add(self.update, "update_motion_controller")

    @torch.no_grad()
    def load_mvae(self, mvae_path: str):
        self.mvae = torch.load(mvae_path)
        self.mvae.eval()

    def _sync_chara(self):
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
            
        root_vel = np.array([vx, 0, vz])

        # Update root joint info.
        root_pos = self.chara.joint_trans[0]
        root_quat = self.chara.joint_orien[0]
        root_rot = R.from_quat(root_quat)

        root_pos = root_pos + root_rot.apply(root_vel) * self.scene.dt
        root_rad_y = QuatUtil.extract_y_rad(root_quat[np.newaxis, ...])
        root_rad_y = root_rad_y.squeeze() + wy * self.scene.dt
        root_quat = R.from_euler("Y", root_rad_y, degrees=False).as_quat()

        # Update joint info.
        joint_pos = np.zeros((self.chara.num_joints, 3))
        joint_pos[1:] = self.chara.offsets[1:]
        joint_pos[0] = root_pos

        joint_orien = np.zeros_like(local_joint_orien)
        joint_orien[..., 3] = 1.
        for i, p in enumerate(self.chara.parent_idx):
            if p == -1: continue
            Rp = R.from_quat(local_joint_orien[p])
            Ri = Rp.inv() * R.from_quat(local_joint_orien[i])
            joint_orien[i] = Ri.as_quat()
        joint_orien[0, :] = root_quat

        # Forward kinematics and sync. to character
        joint_trans, joint_orien = self.chara.bvh.batch_fk(
            joint_pos[np.newaxis, ...],
            joint_orien[np.newaxis, ...]
        )
        self.chara.node.set_x(root_pos[0])
        self.chara.node.set_z(root_pos[2])
        self.chara.joint_trans = joint_trans.squeeze()
        self.chara.joint_orien = joint_orien.squeeze()

    @torch.no_grad()
    def _update_motion(self):
        # Call MVAE to predict motion
        base_z = torch.normal(
            mean=0, std=1,
            size=(1, self.mvae.latent_size)
        ).to("cuda:0")

        self.z = (base_z + self.z) / 2
        c = torch.from_numpy(
            self.motion
        ).float().reshape(1, -1).to("cuda:0")

        c = self.mvae.normalize(c)
        output = self.mvae.sample(
            self.z, c, deterministic=True
        )
        output = self.mvae.denormalize(output).cpu()

        output = output.detach().numpy()
        self.motion = output.squeeze()

    def _sync_controller(self):
        # Sync. controller to character
        self.chara.node.setX(self.root_pos[0])
        self.chara.node.setZ(self.root_pos[2])

    def update(self, task):
        self._sync_chara()
        self._update_motion()
        # self._sync_controller()

        return task.cont
