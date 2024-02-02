from direct.showbase.DirectObject import DirectObject
import numpy as np
from scipy.spatial.transform import Rotation as R

from scene.Scene import Scene
from mocon.motion.BVHMotion import BVHMotion
from mocon.motion.utils.QuatHelper import QuatHelper

class Character:
    def __init__(
        self,
        model: DirectObject,
        scene: Scene,
        bvh_path: str
    ):
        self.model = model
        self.scene = scene

        self.bvh_motion = BVHMotion(bvh_path)
        self.load_bvh_info(frame_idx=300)

        self.scene.taskMgr.add(self.update, "update_chara")

    @property
    def root_pos(self):
        return self.joint_trans[0]

    def load_bvh_info(self, frame_idx: int):
        self.dt = self.bvh_motion.dt
        self.num_joints = self.bvh_motion.num_joints
        self.joint_names = self.bvh_motion.joint_names
        self.parent_idx = self.bvh_motion.parent_idx
        self.offsets = self.bvh_motion.offsets
        self.channels = self.bvh_motion.channels
        self.frame_idx = frame_idx

        # Initialize joint trans. and orien. at a certain frame
        joint_pos = self.bvh_motion.joint_pos[frame_idx][None, ...]
        joint_pos[:, 0, [0, 2]] = 0. # Move the root joint to over the origin
        joint_quat = self.bvh_motion.joint_quat[frame_idx][None, ...]
        joint_trans, joint_orien = self.bvh_motion.batch_fk(
            joint_pos, joint_quat
        )
        self.joint_trans = joint_trans.squeeze()
        self.joint_orien = joint_orien.squeeze()

    def sync(self, sync_motion: dict):
        # Update character state with formatted motion info.
        root_vel = sync_motion["root_vel"]
        root_avel_y = sync_motion["root_avel_y"]
        local_joint_orien = sync_motion["local_joint_orien"]

        root_pos = self.joint_trans[0]
        root_quat = self.joint_orien[0]
        root_rot = R.from_quat(root_quat)

        root_pos = root_pos + root_rot.apply(root_vel) * self.dt
        root_angle_y = QuatHelper.extract_y_rad(root_quat[None, ...])
        root_angle_y = root_angle_y.squeeze() + root_avel_y * self.dt
        root_quat = R.from_euler("Y", root_angle_y, degrees=False).as_quat()

        # jointPos & jointRot
        joint_pos = np.zeros((self.num_joints, 3))
        joint_pos[1:] = self.offsets[1:]
        joint_pos[0] = root_pos

        joint_quat = np.zeros_like(local_joint_orien)
        joint_quat[..., 3] = 1.
        for i, p in enumerate(self.parent_idx):
            if p == -1: continue
            Rp = R.from_quat(local_joint_orien[p])
            Ri = Rp.inv() * R.from_quat(local_joint_orien[i])
            joint_quat[i] = Ri.as_quat()
        joint_quat[0, :] = root_quat

        # forward kinematics
        joint_trans, joint_orien = self.bvh_motion.batch_fk(
            joint_pos[None, ...],
            joint_quat[None, ...]
        )
        self.joint_trans = joint_trans.squeeze()
        self.joint_orien = joint_orien.squeeze()

    def update(self, task):
        self._update_model(
            self.joint_names,
            self.joint_trans,
            self.joint_orien
        )
        return task.cont

    def _update_model(
        self,
        joint_names,
        joint_trans,
        joint_orien
    ):
        # Update joint position and orientation in a frame
        for i in range(len(joint_names)):
            name = joint_names[i]
            self.model.set_joints(
                name, joint_trans[i], joint_orien[i]
            )

    def _getJointPos(self):
        jointPos = [
            joint.getPos() for
            joint in self.model.getJoints()
        ]
        jointPos = np.array(jointPos)
        return jointPos
    
    def _getJointQuat(self):
        jointQuat = [
            joint.getQuat() for
            joint in self.model.getJoints()
        ]
        jointQuat = np.array(jointQuat)
        return jointQuat