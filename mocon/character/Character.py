import numpy as np
from scipy.spatial.transform import Rotation as R

from scene.Scene import Scene
from mocon.motion.BVHMotion import BVHMotion
from mocon.motion.utils.QuatUtil import QuatUtil

class Character:
    def __init__(
        self,
        scene: Scene,
        model,
        use_mvae=True
    ):
        self.model = model
        self.scene = scene

        self.node = self.scene.render.attach_new_node("chara")
        self.node.set_pos(self.scene.render, 0, 0.01, 0)

        if use_mvae:
            # Load BVH file and init mvae pose
            self.frame = 300
            self.bvh_file = "mocon/motion/mocap/bvh/walk1_subject5.bvh"
            self._load_bvh(self.bvh_file, self.frame)
            self.npz_file = "mocon/motion/mocap/npz/walk1_subject5.npz"
            self.mvae_path = "mocon/motion/mvae/model/walk1_subject5_240129_190220.pt"
            
        else:
            pass
            # use state machine to control character motion

        # self.root_pos = self.node.get_pos()
        self.scene.task_mgr.add(self.update, "update_chara")

    @property
    def x(self):
        return self.node.get_x()
    
    @property
    def z(self):
        return self.node.get_z()
    
    def _load_bvh(self, bvh_file, frame):
        self.bvh = BVHMotion(bvh_file)
        # static info.
        self.joint_names = self.bvh.joint_names
        self.num_joints = self.bvh.num_joints
        self.parent_idx = self.bvh.parent_idx
        self.offsets = self.bvh.offsets
        self.channels = self.bvh.channels
        # dynamic info.
        joint_quat = self.bvh.joint_quat[frame][np.newaxis, ...]
        joint_pos = self.bvh.joint_pos[frame][np.newaxis, ...]
        joint_pos[:, 0, [0,2]] = self.x, self.z
        # forward kinematics
        joint_trans, joint_orien = self.bvh.batch_fk(joint_pos, joint_quat)
        self.joint_trans = joint_trans.squeeze()
        self.joint_orien = joint_orien.squeeze()

    def _update_model(self):
        for i in range(self.num_joints):
            self.model.set_joints(
                self.joint_names[i],
                self.joint_trans[i],
                self.joint_orien[i]
            )

    def update(self, task):
        self._update_model()
        return task.cont
    