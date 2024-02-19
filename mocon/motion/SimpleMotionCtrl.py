import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.character.Character import Character
from mocon.motion.mocap.BVHMotion import BVHMotion
from mocon.utils.MocapUtil import MocapUtil
from mocon.utils.QuatUtil import QuatUtil
from mocon.utils.SpringUtil import SpringUtil

class SimpleMotionCtrl:
    def __init__(self, chara: Character, chara_ctrl):
        self.chara = chara
        self.scene = self.chara.scene
        self.chara_ctrl = chara_ctrl

        self.motions = []
        self.motions.append(BVHMotion("mocon/motion/mocap/states/walk_forward.bvh"))
        self.motions.append(BVHMotion("mocon/motion/mocap/states/idle.bvh"))
        self.motion_id = 0
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0

        self.walk = MocapUtil.build_loop_motion(self.motions[0])
        self.idle = MocapUtil.build_loop_motion(self.motions[1])
        self.idle2move = MocapUtil.cat_two_clips(self.motions[1], self.motions[0], 60, 30)
        self.move2idle = MocapUtil.cat_two_clips(self.motions[0], self.motions[1], 60, 30)
        self.cur_state = "idle"

        self.scene.task_mgr.add(self.update, "update_motion")
    
    def _update_motion(
            self,
            desired_pos_list, desired_rot_list,
            desired_vel_list, desired_avel_list,
            current_gait=None
        ):
        # motion_id : 0 move , 1 idle
        last_state = self.cur_state
        if abs(desired_vel_list[0,0]) + abs(desired_vel_list[0,1]) < 1e-2:
            self.cur_state = "idle"
        else:
            self.cur_state = "move"

        if self.cur_state == "move":
            motion_id = self.motion_id
            cur_motion = self.walk.raw_copy()

            if self.cur_state != last_state:
                target_orien_xz = R.from_quat(
                    self.idle.joint_quat[self.cur_frame, 0, :]
                ).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                cur_motion = cur_motion.retarget_root(
                    0,
                    self.idle.joint_pos[self.cur_frame, 0, [0, 2]],
                    target_orien_xz
                )

                self.cur_frame = 0

            key_frame = [
                (self.cur_frame + i*20) % self.motions[motion_id].motion_length
                for i in range(3)
            ]
            cur_motion_key_frame_vel = cur_motion.joint_pos[key_frame, 0, :] - \
                cur_motion.joint_pos[[(frame - 1) for frame in key_frame], 0, :]
            current_motion_avel = QuatUtil.quat2avel(cur_motion.joint_quat[:, 0, :], 1/60)

            # only for root joint
            diff_root_pos = desired_pos_list - cur_motion.joint_pos[key_frame, 0, :]
            diff_root_pos[:, 1] = 0
            diff_root_rot = (
                R.from_quat(desired_rot_list[0:3]) * R.from_quat(cur_motion.joint_quat[key_frame, 0, :]).inv()
            ).as_rotvec()
            diff_root_vel = (desired_vel_list - cur_motion_key_frame_vel) / 60
            diff_root_avel = desired_avel_list[0:6] - current_motion_avel[[(frame-1) for frame in key_frame]]

            for i in range(self.cur_frame, self.cur_frame+self.motions[motion_id].motion_length//2):
                halflife = 0.2
                index = (i - self.cur_frame) // 20
                dt = (i-self.cur_frame) % 20

                off_pos, _ = SpringUtil.decay_spring_implicit_damping_pos(
                    diff_root_pos[index], diff_root_vel[index], halflife, dt/60
                )
                off_rot, _ = SpringUtil.decay_spring_implicit_damping_rot(
                    diff_root_rot[index], diff_root_avel[index], halflife, dt/60
                )

                cur_motion.joint_pos[i % self.motions[motion_id].motion_length, 0, :] += off_pos
                cur_motion.joint_quat[i % self.motions[motion_id].motion_length, 0, :] = (
                    R.from_rotvec(off_rot) * R.from_quat(
                        cur_motion.joint_quat[i % self.motions[motion_id].motion_length, 0, :]
                    )
                ).as_quat()

            joint_trans, joint_orien = cur_motion.batch_fk()
            joint_trans = joint_trans[self.cur_frame]
            joint_orien = joint_orien[self.cur_frame]
            self.cur_root_pos = joint_trans[0]
            self.cur_root_rot = joint_orien[0]

            self.walk = cur_motion
            self.cur_frame = (self.cur_frame + 1) % self.motions[motion_id].motion_length

        elif self.cur_state == "idle":
            motion_id = self.motion_id
            cur_motion = self.idle
            if self.cur_state != last_state:
                target_orien_xz = R.from_quat(
                    self.walk.joint_quat[self.cur_frame, 0, :]
                ).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                cur_motion = cur_motion.retarget_root(
                    0,
                    self.walk.joint_pos[self.cur_frame, 0, [0, 2]],
                    target_orien_xz
                )
                self.cur_frame = 0

            joint_trans, joint_orien = cur_motion.batch_fk()
            joint_trans = joint_trans[self.cur_frame]
            joint_orien = joint_orien[self.cur_frame]
            self.cur_root_pos = joint_trans[0]
            self.cur_root_rot = joint_orien[0]
            self.cur_frame = 0
            self.idle = cur_motion
            self.cur_frame = (self.cur_frame + 1) % self.motions[motion_id].motion_length

        return joint_trans, joint_orien

    def _sync_chara(self):
        self.chara.node.set_x(self.cur_root_pos[0])
        self.chara.node.set_z(self.cur_root_pos[2])

    def update(self, task):
        (
            desired_pos_list,
            desired_rot_list,
            desired_vel_list,
            desired_avel_list
        ) = self.chara_ctrl.get_desired_state()

        (
            self.chara.joint_trans,
            self.chara.joint_orien
        ) = self._update_motion(
            desired_pos_list, 
            desired_rot_list,
            desired_vel_list,
            desired_avel_list
        )

        self._sync_chara()

        return task.cont
