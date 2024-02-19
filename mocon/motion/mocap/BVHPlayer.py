from mocon.character.Character import Character
from mocon.motion.mocap.BVHMotion import BVHMotion

class BVHPlayer:
    def __init__(
        self,
        chara: Character,
        bvh_file: str,
        npz_file = None
    ):
        self.chara = chara
        self.scene = chara.scene

        self.bvh = BVHMotion(bvh_file)
        self.cur_frame = 0
        self.joint_names = self.bvh.joint_names

        joint_pos, joint_quat = None, None
        if npz_file is not None:
            joint_pos, joint_quat = self.bvh.load_mvae_mocap(npz_file)
        self.joint_trans, self.joint_orien = self.bvh.batch_fk(
            joint_pos, joint_quat
        )

        self.scene.task_mgr.add(self.update, "update_bvh_player")
        
    def update(self, task):
        speed_inv = 1
        for i in range(len(self.joint_names)):
            self.chara.model.set_joints(
                self.joint_names[i],
                self.joint_trans[self.cur_frame//speed_inv, i, :],
                self.joint_orien[self.cur_frame//speed_inv, i, :]
            )

        self.chara.node.set_x(self.joint_trans[self.cur_frame//speed_inv, 0, 0])
        self.chara.node.set_z(self.joint_trans[self.cur_frame//speed_inv, 0, 2])

        self.cur_frame = (self.cur_frame + 1) % (self.joint_trans.shape[0]*speed_inv)
        return task.cont