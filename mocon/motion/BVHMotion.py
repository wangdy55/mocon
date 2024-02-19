import os
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.utils.QuatUtil import QuatUtil

"""
N = numFrames
M = numJoints
"""

class BVHMotion():
    def __init__(self, bvh_path: str):
        self.bvh_path = bvh_path

        # static data
        self.joint_names = []
        self.parent_idx = []
        self.channels = []
        self.offsets = []
        self.dt = 0.01

        # motion data
        self.start_frame = 0
        self.num_frames = 0
        self.frame_size = 0
        self.joint_pos = None  # shape = (N, M, 3)
        self.joint_quat = None  # shape = (N, M, 4), R.quat

        if self.bvh_path is not None:
            self._load_bvh()
        pass

    @property
    def num_joints(self):
        return len(self.joint_names)
    
    @property
    def motion_length(self):
        return self.num_frames

    def _load_static_data(self) -> tuple:
        with open(self.bvh_path, "r") as f:
            joint_names = []
            parent_idx = []
            channels = []
            offsets = []
            end_sites = []
            dt = 0

            parent_stack = [None]
            for line in f:
                if "ROOT" in line or "JOINT" in line:
                    joint_names.append(line.split()[-1])
                    parent_idx.append(parent_stack[-1])
                    channels.append("")
                    offsets.append([0, 0, 0])

                elif "End Site" in line:
                    end_sites.append(len(joint_names))
                    joint_names.append(parent_stack[-1] + "_end")
                    parent_idx.append(parent_stack[-1])
                    channels.append("")
                    offsets.append([0, 0, 0])

                elif "{" in line:
                    parent_stack.append(joint_names[-1])

                elif "}" in line:
                    parent_stack.pop()

                elif "OFFSET" in line:
                    offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1, 3)

                elif "CHANNELS" in line:
                    pos_order = []
                    rot_order = []
                    for token in line.split():
                        if "position" in token:
                            pos_order.append(token[0])

                        if "rotation" in token:
                            rot_order.append(token[0])

                    channels[-1] = "".join(pos_order) + "".join(rot_order)

                elif "Frame Time:" in line:
                    dt = float(line.split()[-1])
                    break

        parent_idx = [-1] + [joint_names.index(i) for i in parent_idx[1:]]
        channels = [len(i) for i in channels]
        offsets = np.concatenate(offsets, axis=0)
        return joint_names, parent_idx, channels, offsets, dt

    def _load_motion_data(self, start_frame=0) -> np.ndarray:
        """
        @Parameters:
            start_frame: int, which frame to start from
        
        @Returns:
            motion_data: ndarray, shape = (N, frame_size)
        """
        with open(self.bvh_path, "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].startswith("Frame Time"):
                    break
            motion_data = []
            for line in lines[i+1:]:
                data = [float(x) for x in line.split()]
                if len(data) == 0:
                    break
                motion_data.append(np.array(data).reshape(1, -1))
            motion_data = np.concatenate(motion_data, axis=0)
        self.start_frame = start_frame
        return motion_data[start_frame:]

    def _load_bvh(self):
        self.joint_names, self.parent_idx, self.channels, self.offsets, self.dt = self._load_static_data()
        motion_data = self._load_motion_data()

        self.num_frames = motion_data.shape[0]
        self.frame_size = motion_data.shape[1]

        self.joint_pos = np.zeros([self.num_frames, self.num_joints, 3])
        self.joint_quat = np.zeros([self.num_frames, self.num_joints, 4])
        self.joint_quat[:, :, 3] = 1.0

        channel_idx = 0
        for i in range(len(self.joint_names)):
            if self.channels[i] == 0:
                self.joint_pos[:, i] = self.offsets[i].reshape(1, 3)
                continue

            if self.channels[i] == 3:
                position = self.offsets[i].reshape(1, 3)
                rotation = motion_data[:, channel_idx:channel_idx+3]
            elif self.channels[i] == 6:
                position = motion_data[:, channel_idx:channel_idx+3]
                rotation = motion_data[:, channel_idx+3:channel_idx+6]
            self.joint_pos[:, i, :] = position
            self.joint_quat[:, i, :] = R.from_euler("XYZ", rotation, degrees=True).as_quat()

            channel_idx += self.channels[i]

        return

    def batch_fk(self, joint_pos=None, joint_quat=None) -> tuple:
        """
        batch forward kinematics

        @Parameters:
            joint_pos: ndarray, shape = (N, M, 3)
            joint_rot: ndarray, shape = (N, M, 4), R.quat
        
        @Returns:
            jointTrans: ndarray, shape = (N, M, 3)
            jointOrien: ndarray, shape = (N, M, 4), R.quat
        """
        if joint_pos is None:
            joint_pos = self.joint_pos
        if joint_quat is None:
            joint_quat = self.joint_quat

        joint_trans = np.zeros_like(joint_pos)
        joint_orien = np.zeros_like(joint_quat)
        joint_orien[..., 3] = 1.0

        for i, p in enumerate(self.parent_idx):
            if p == -1:
                joint_trans[:, i] = joint_pos[:, i]
                joint_orien[:, i] = joint_quat[:, i]
            else:
                Op = R.from_quat(joint_orien[:, p])
                Oi = Op * R.from_quat(joint_quat[:, i])
                joint_trans[:, i] = joint_trans[:, p] + Op.apply(joint_pos[:, i])
                joint_orien[:, i] = Oi.as_quat()

        return joint_trans, joint_orien
   
    def _adjust_joint_names(self, targetJointNames):
        """
        Adjust joint seq. to target joint names
        """
        idx = [self.joint_names.index(name) for name in targetJointNames]
        idx_inv = [targetJointNames.index(name) for name in self.joint_names]
        self.joint_names = [self.joint_names[i] for i in idx]
        self.parent_idx = [idx_inv[self.parent_idx[i]] for i in idx]
        self.parent_idx[0] = -1
        self.joint_channel = [self.channels[i] for i in idx]
        self.joint_pos = self.joint_pos[:,idx,:]
        self.joint_quat = self.joint_quat[:,idx,:]
    
    def raw_copy(self):
        return copy.deepcopy(self)
    
    def _sub_seq(self, start, end):
        """
        Get a sub sequence info. of joint position and rotation

        @Parameters:
            start, end: int, frame index

        @Returns:
            res: sub sequence
        """
        res = self.raw_copy()
        res.joint_pos = res.joint_pos[start:end,:,:]
        res.joint_quat = res.joint_quat[start:end,:,:]
        return res
    
    def _append(self, other):
        """
        Append a motion to the end
        """
        other = other.raw_copy()
        other.adjust_joint_names(self.joint_names)
        self.joint_pos = np.concatenate(
            (self.joint_pos, other.joint_pos),
            axis=0
        )
        self.joint_quat = np.concatenate(
            (self.joint_quat, other.joint_quat), 
            axis=0
        )
    
    def retarget_root(self, frame_idx, target_trans_xz, target_orien_xz):
        res = self.raw_copy()
        
        offset = target_trans_xz - res.joint_pos[frame_idx, 0, [0,2]]
        res.joint_pos[:, 0, [0,2]] += offset
        
        # \theta between target_orien_xz and z-axis -> y-axis angle \rho
        sin_theta = np.cross(target_orien_xz, [0, 1]) / np.linalg.norm(target_orien_xz)
        theta = np.arcsin(sin_theta) # rad
        theta = theta if theta > 0 else 2 * np.pi - theta
        rho = R.from_euler('Y', theta)
        # Decompose quat to y comp.
        q_y, _ = QuatUtil.y_axis_decompose(res.joint_quat[frame_idx, 0, :])
        r_y = R(q_y)

        # root.Ry => its y-axis component
        r = R.from_quat(res.joint_quat[:, 0, :])
        res.joint_quat[:, 0, :] = (rho * r_y.inv() * r).as_quat()

        # root_pos = \rho.apply(root_pos)
        v_xz =  res.joint_pos[:, 0, :] - res.joint_pos[frame_idx, 0, :]
        v_xz = (rho * r_y.inv()).apply(v_xz)
        res.joint_pos[:, 0, :] = res.joint_pos[frame_idx, 0, :] + v_xz

        return res

    # for MotionVAE motion format
    def get_local_joint_info(self) -> tuple:
        """
        Get local (root based) joint info. in MotionVAE format

        @Return:
            local joint info: tuple, (Jp, Jv, Jr)
        """
        # local batch FK -> local joint position (Jp)
        local_joint_pos = self.joint_pos.copy()
        local_joint_pos[:, 0] = [0., 0., 0.]
        local_joint_quat = self.joint_quat.copy()
        local_joint_quat[:, 0] = [0., 0., 0., 1.]
        local_joint_trans, local_joint_orien = self.batch_fk(
            joint_pos=local_joint_pos,
            joint_quat=local_joint_quat
        )

        # Remove end effector info.
        not_ee = [i for i, ch in enumerate(self.channels) if ch != 0]
        local_joint_trans = local_joint_trans[:, not_ee]
        local_joint_orien = local_joint_orien[:, not_ee]

        # Jp -> Jv, shape = (N-1, M, 3)
        local_joint_vel = (local_joint_trans[1:] - local_joint_trans[:-1]) / self.dt

        # local orientation -> local joint vec6d (Jr)
        # vec6d: first 2 col (xy) of local orientation matrix
        local_joint_mat = np.zeros([self.num_frames, len(not_ee), 3, 3])
        for i in range(len(not_ee)):
            local_joint_mat[:, i] = R.from_quat(local_joint_orien[:, i]).as_matrix()
        local_joint_vet6d = local_joint_mat[..., :2]
        local_joint_vet6d = local_joint_vet6d.reshape(self.num_frames, len(not_ee), -1)

        return local_joint_trans[:-1], local_joint_vel, local_joint_vet6d[:-1]

    def save_mvae_mocap(self, npz_path: str):
        """
        Convert raw motion data to
        MVAE motion format:
            (r'x, r'z, w_y, Jp, Jv, Jr)
            Jx = (jx_1, jx_2, ..., jx_m), x = {p, v, r}
        and save as npz file

        @Parameters:
            mvae_motion_file_path: str, path to MVAE motion npz file

        @Returns:
        """
        if os.path.exists(npz_path):
            print(f"{npz_path} already exists, skip saving")
            return

        # root info: (r'x, r'z, w_y)
        root_quat = self.joint_quat[:, 0].copy()
        root_pos = self.joint_pos[:, 0].copy()

        # root orientation -> w_y (rad/s)
        root_angle_y = QuatUtil.extract_y_rad(root_quat)
        root_avel_y = root_angle_y[1:] - root_angle_y[:-1]
        # Fix rotation around ±π
        root_avel_y[root_avel_y >= np.pi] -= 2 * np.pi
        root_avel_y[root_avel_y <= -np.pi] += 2 * np.pi
        root_avel_y /= self.dt

        # root translation -> r'x, r'z
        root_vel = (root_pos[1:, :3] - root_pos[:-1, :3]) / self.dt # shape = (N-1, 3)
        # global velocity -> local velocity
        root_rot = R.from_quat(root_quat)[:-1]
        root_vel_xz = root_rot.apply(root_vel, inverse=True)[:, [0, 2]]
        
        # local batch FK -> Jp, Jv, Jr
        local_joint_trans, local_joint_vel, local_joint_vec6d = self.get_local_joint_info()
        num_not_ee = local_joint_trans.shape[1]

        # Remove root info. and reshape
        local_joint_trans = local_joint_trans[:, 1:].reshape(self.num_frames-1, -1)
        local_joint_vel = local_joint_vel[:, 1:].reshape(self.num_frames-1, -1)
        local_joint_vec6d = local_joint_vec6d[:, 1:].reshape(self.num_frames-1, -1)

        # Cat. -> (v_x, v_z, w_y(1), Jp(19*3), Jv(19*3), Jr(19*6)), shape = (N-1, frame_size=3+228)
        mvae_motion = np.concatenate(
            (root_vel_xz, root_avel_y.reshape(-1, 1), local_joint_trans, local_joint_vel, local_joint_vec6d),
            axis=1
        )

        np.savez(
            npz_path,
            nun_frames=self.num_frames,
            num_joints=self.num_joints,
            num_not_ee=num_not_ee,
            root_angle_y_start=root_angle_y[0],
            root_pos_start=root_pos[0],
            mvae_motion=mvae_motion
        )
        print("MVAE mocap saved to", npz_path)

    def load_mvae_mocap(self, npz_path:str) -> tuple:
        """
        Recover BVH mocap from MVAE mocap

        @Parameters:
            npz_path: str, path to local_motion.npz

        @Returns:
            joint_pos: ndarray, shape = (N, M, 3)
            joint_quat: ndarray, shape = (N, M, 4), R.quat
        """

        # mvae motion -> joint position, joint rotation, for batch FK
        mvae_mocap = np.load(npz_path)
        num_frames = mvae_mocap["num_frames"]
        num_joints = mvae_mocap["num_joints"]
        num_not_ee = mvae_mocap["num_not_ee"]
        root_angle_y_start = mvae_mocap["root_angle_y_start"]
        root_pos_start = mvae_mocap["root_pos_start"]
        mvae_motion = mvae_mocap["mvae_motion"]
        # MVAE motion sliced
        root_vel_xz = mvae_motion[:, :2]
        root_avel_y = mvae_motion[:, 2]
        local_joint_vec6d = mvae_motion[:, -(num_not_ee-1)*6:]

        # root rotation
        root_angle_y = np.zeros([num_frames])
        root_angle_y[0] = root_angle_y_start
        for i in range(1, num_frames):
            root_angle_y[i] = root_angle_y[i-1] + root_avel_y[i-1] * self.dt
        root_angle_y = root_angle_y.reshape(num_frames, 1)
        root_rot = R.from_euler("Y", root_angle_y[:-1], degrees=False)
        root_quat = root_rot.as_quat()

        # root position
        zero_vel = np.zeros([num_frames-1, 1])
        root_vel_x = root_vel_xz[:, 0].reshape(-1, 1)
        root_vel_z = root_vel_xz[:, 1].reshape(-1, 1)
        root_vel = np.concatenate((root_vel_x, zero_vel, root_vel_z), axis=1)

        root_pos = np.zeros([num_frames-1, 3])
        root_pos[0] = root_pos_start
        for i in range(1, num_frames-1):
            root_pos[i] = root_pos[i-1] + root_rot[i-1].apply(root_vel[i-1]) * self.dt

        # local joint orien. -> joint rotation
        joint_orien = np.zeros([num_frames-1, num_joints, 4])
        joint_orien[..., 3] = 1.
        local_joint_vec6d = local_joint_vec6d.reshape(num_frames-1, num_not_ee-1, 3, 2)
        local_joint_z = np.cross(
            local_joint_vec6d[..., 0], local_joint_vec6d[..., 1], axis=-1
        ).reshape(num_frames-1, num_not_ee-1, 3, 1)
        local_joint_z /= np.linalg.norm(local_joint_z, axis=-2, keepdims=True)
        local_joint_mat = np.concatenate((local_joint_vec6d, local_joint_z), axis=-1)

        j = 0
        for i in range(1, num_joints):
            if (self.channels[i] == 0): # Skip end effectors
                continue
            joint_orien[:, i] = R.from_matrix(local_joint_mat[:, j]).as_quat()
            j += 1

        # joint position & rotation
        joint_pos = np.zeros((num_frames-1, num_joints, 3))
        joint_pos[:, 1:, :] = np.tile(self.offsets[1:], (self.num_frames-1, 1, 1))
        joint_pos[:, 0] = root_pos

        joint_quat = np.zeros_like(joint_orien)
        joint_quat[..., 3] = 1.
        for i, p in enumerate(self.parent_idx):
            if p == -1: continue
            Rp = R.from_quat(joint_orien[:, p])
            Ri = Rp.inv() * R.from_quat(joint_orien[:, i])
            joint_quat[:, i] = Ri.as_quat()
        joint_quat[:, 0, :] = root_quat

        return joint_pos, joint_quat
