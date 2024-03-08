import numpy as np

from scene.Scene import Scene
from mocon.motion.mocap.BVHMotion import BVHMotion

bvh_filename = "walk1_subject5"
mvae_filename = "walk1_subject5_240129_190220"

class Character:
    def __init__(
        self,
        scene: Scene,
        model,
        random
    ):
        self.model = model
        self.scene = scene

        self.node = self.scene.render.attach_new_node("chara")
        self.node.set_pos(self.scene.render, 0, 0.01, 0)

        # Load BVH file and init mvae pose
        self.frame = 300
        self.bvh_file = f"mocon/motion/mocap/bvh/{bvh_filename}.bvh"
        self._load_bvh(self.bvh_file, self.frame)
        self.npz_file = f"mocon/motion/mocap/npz/{bvh_filename}.npz"
        self.mvae_path = f"mocon/motion/mvae/model/{mvae_filename}.pt"

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
    