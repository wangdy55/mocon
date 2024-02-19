from scipy.spatial.transform import Rotation as R 
import numpy as np

from mocon.utils.QuatUtil import QuatUtil
from mocon.utils.SpringUtil import SpringUtil

class MocapUtil:
    @staticmethod
    def cat_two_clips(
        bvh1,
        bvh2,
        mix_frame_idx,
        mix_time
    ):
        res = bvh1.raw_copy()

        # alignment: from mix_frame1 to the 1st frame of new motion
        root_quat = bvh1.joint_quat[mix_frame_idx, 0]
        facing_axis = R.from_quat(root_quat).apply(
            np.array([0, 0, 1])
        ).flatten()[[0, 2]]
        new_bvh2 = bvh2.retarget_root(
            0,
            bvh1.joint_pos[mix_frame_idx, 0, [0,2]],
            facing_axis
        )

        # inertial interpolation
        halflife = 0.27
        d = SpringUtil.halflife2dampling(halflife)
        dt = 1/100
        
        src_avel = QuatUtil.quat2avel(
            bvh1.joint_quat[mix_frame_idx-10:mix_frame_idx], dt
        )
        tar_avel = QuatUtil.quat2avel(
            new_bvh2.joint_quat[0:10], dt
        )
        delta_avel = src_avel[-1] - tar_avel[0]
        delta_rotvec = (
            R.from_quat(bvh1.joint_quat[mix_frame_idx]) * R.from_quat(new_bvh2.joint_quat[0].copy()).inv()
        ).as_rotvec()

        src_vel = bvh1.joint_pos[mix_frame_idx] - bvh1.joint_pos[mix_frame_idx-1]
        tar_vel = new_bvh2.joint_pos[1] - new_bvh2.joint_pos[0]
        delta_vel = (src_vel - tar_vel) / 60
        delta_pos = bvh1.joint_pos[mix_frame_idx] - new_bvh2.joint_pos[0]

        for i in range(len(new_bvh2.joint_pos)):
            tmp_ydt = d * i * dt
            eydt = np.exp( -tmp_ydt)
            j1 = delta_vel + delta_pos * d
            j2 = delta_avel + delta_rotvec * d
            off_pos_i = eydt * (delta_pos + j1 * i * dt)
            off_rot_i = R.from_rotvec(eydt * (delta_rotvec + j2 * i * dt)).as_rotvec()

            new_bvh2.joint_pos[i] = new_bvh2.joint_pos[i] + off_pos_i
            new_bvh2.joint_quat[i] = (
                R.from_rotvec(off_rot_i) * R.from_quat(new_bvh2.joint_quat[i])
            ).as_quat()

        res.joint_pos = np.concatenate([res.joint_pos[:mix_frame_idx], new_bvh2.joint_pos], axis=0)
        res.joint_quat = np.concatenate([res.joint_quat[:mix_frame_idx], new_bvh2.joint_quat], axis=0)
        
        return res
    
    @staticmethod
    def build_loop_motion(bvh_motion, halflife=0.2, fps=60):
        # Process rotations
        quat = bvh_motion.joint_quat
        avel = QuatUtil.quat2avel(quat, 1/60)

        # Calculate rotation difference between the last and first frames
        rot_diff = (
            R.from_quat(quat[-1]) * R.from_quat(quat[0].copy()).inv()
        ).as_rotvec()
        avel_diff = (avel[-1] - avel[0])

        # Evenly distribute the rotation difference to each frame
        for i in range(bvh_motion.motion_length):
            offset1 = SpringUtil.decay_spring_implicit_damping_rot(
                0.5*rot_diff, 0.5*avel_diff, halflife, i/fps
            )
            offset2 = SpringUtil.decay_spring_implicit_damping_rot(
                -0.5*rot_diff, -0.5*avel_diff, halflife, (bvh_motion.motion_length-i-1)/fps
            )
            offset_rot = R.from_rotvec(offset1[0] + offset2[0])
            bvh_motion.joint_quat[i] = (offset_rot * R.from_quat(quat[i])).as_quat() 
        
        # Process positions
        pos_diff = bvh_motion.joint_pos[-1] - bvh_motion.joint_pos[0]
        pos_diff[:, [0,2]] = 0
        vel1 = bvh_motion.joint_pos[-1] - bvh_motion.joint_pos[-2]
        vel2 = bvh_motion.joint_pos[1] - bvh_motion.joint_pos[0]
        vel_diff = (vel1 - vel2) / 60
        
        for i in range(bvh_motion.motion_length):
            offset1 = SpringUtil.decay_spring_implicit_damping_pos(
                0.5*pos_diff, 0.5*vel_diff, halflife, i/fps
            )
            offset2 = SpringUtil.decay_spring_implicit_damping_pos(
                -0.5*pos_diff, -0.5*vel_diff, halflife, (bvh_motion.motion_length-i-1)/fps
            )
            offset_pos = offset1[0] + offset2[0]
            bvh_motion.joint_pos[i] += offset_pos

        bvh_motion.joint_pos[:, 0, [0,2]] = bvh_motion.joint_pos[0, 0, [0,2]]
        
        return bvh_motion