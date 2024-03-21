import panda3d.core as p3d
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.controller.CameraCtrl import CameraCtrl
from mocon.utils.SpringUtil import SpringUtil
from mocon.utils.ShowUtil import ShowUtil

from mocon.character.Character import Character

class CharacterCtrl:
    def __init__(
        self,
        chara: Character,
        camera_ctrl: CameraCtrl
    ):
        super().__init__()
        self.chara = chara
        self.camera_ctrl = camera_ctrl

        self.scene = self.chara.scene
        self.scene.task_mgr.add(self.update, "update_chara_ctrl")
        self.dt = self.scene.dt

        self.vel = p3d.LVector3(0, 0, 0)
        self.acc = p3d.LVector3(0, 0, 0)
        self.avel = p3d.LVector3(0, 0, 0)
        self.user_input = p3d.LVector3(0, 0, 0) # user input in camera forward direction

        # props of future track
        self.future_wind = 3
        self.sub_step = 20
        self.future_nodes = []
        self.future_pos = []
        self.future_rot = []
        self.future_vel = []
        self.future_avel = []
        self.future_acc = []

        self.next_pos = p3d.LVector3(0, 0, 0)
        self.next_rot = np.array([0, 0, 0, 1])
        self.dv = p3d.LVector3(0, 0, 0)
        self.dr = p3d.LVector3(0, 0, 0)
        self.halflife = 0.27
        self.move_speed = p3d.LVector3(1.0, 1.0, 1.0)

        self._set_key_map()
        # self._init_key_input()

        self.node = self.chara.node
        arrow_color = [0, 1, 0, 1]
        ShowUtil.draw_marker(self.node, color=arrow_color)
        self.future_nodes.append(self.node)
        for i in range(1, self.future_wind):
            node = self.scene.render.attach_new_node(f"future_node{i}")
            node.set_pos(0, 0.01, 0)
            ShowUtil.draw_marker(node, color=arrow_color)
            node.reparent_to(self.scene.render)
            self.future_nodes.append(node)
    
    @property
    def rotation(self):
        return np.array(self.node.get_quat())[[1, 2, 3, 0]]
    
    @property
    def position(self):
        return self.node.get_pos()
    
    def _handle_input(self, axis, val):
        if axis == "x":
            self.user_input[0] = val
        elif axis == "z":
            self.user_input[2] = val

    def _set_key_map(self):
        key_map = {
            ("w", "arrow_up"): ["z", 1],
            ("s", "arrow_down"): ["z", -1],
            ("a", "arrow_left"): ["x", 1],
            ("d", "arrow_right"): ["x", -1],
        }
        for keys, vals in key_map.items():
            # wasd
            self.scene.accept(keys[0], self._handle_input, vals)
            self.scene.accept(keys[0]+"-up", self._handle_input, [vals[0], 0])
            # arrows
            self.scene.accept(keys[1], self._handle_input, vals)
            self.scene.accept(keys[1]+"-up", self._handle_input, [vals[0], 0])
        
    def _init_key_input(self):
        self.cam_ref_node = self.node.attach_new_node("cam_ref_node")
        self.cam_ref_node.set_pos(0, 3, -5)
        self.camera_ctrl.pos = self.cam_ref_node.get_pos(self.scene.render)
        self.cam_ref_node.wrt_reparent_to(self.node)

    def _get_target_vel(
        self,
        cam_fwd: p3d.LVector3,
        user_input: p3d.LVector3,
        chara_rot: np.ndarray
    ) -> np.ndarray:
        # Get global target velocity to update position
        cam_fwd[1] = 0
        fwd_speed, side_speed, back_speed = self.move_speed
        # camera rotation around y-axis
        angle = np.arctan2(cam_fwd[0], cam_fwd[2])
        y_rot = R.from_rotvec(angle * np.array([0, 1, 0]))
        # coordinate transformation from camera to global
        global_input = y_rot.apply(user_input)

        # [coordinate transformation] from global to chara's local
        local_target_direct = R.from_quat(chara_rot).apply(global_input, inverse=True)
        if local_target_direct[2] > 0:
            local_target_vel = np.array([side_speed, 0, fwd_speed]) * local_target_direct
        else:
            local_target_vel = np.array([side_speed, 0, back_speed]) * local_target_direct

        global_target_vel = R.from_quat(chara_rot).apply(local_target_vel)
        return global_target_vel

    def _get_target_rot(self, global_target_vel: np.ndarray) -> np.ndarray:
        if np.linalg.norm(global_target_vel) < 1e-5:
            return self.rotation
        else:
            global_target_direct = global_target_vel / np.linalg.norm(global_target_vel)
            y_rotvec = np.arctan2(
                global_target_direct[0], global_target_direct[2]
            ) * np.array([0, 1, 0])
            return R.from_rotvec(y_rotvec).as_quat()

    def _predict_future_pos(self, cur_pos, desired_vel):
        # pos0 (cur) => pos1 => ... => pos_i
        future_pos = [cur_pos] * self.future_wind
        future_vel = [self.vel] * self.future_wind
        future_acc = [self.acc] * self.future_wind

        for i in range(1, self.future_wind):
            next_pos, next_vel, next_acc = SpringUtil.update_spring_pos(
                future_pos[i-1],
                future_vel[i-1],
                future_acc[i-1],
                desired_vel,
                self.halflife,
                self.dt * self.sub_step
            )
            future_pos[i] = next_pos
            future_vel[i] = next_vel
            future_acc[i] = next_acc

        future_pos = np.array(future_pos).reshape(-1, 3)
        future_vel = np.array(future_vel).reshape(-1, 3)
        future_acc = np.array(future_acc).reshape(-1, 3)
        return future_pos, future_vel, future_acc

    def _predict_future_rot(self, cur_rot, desired_rot):
        # rot0 (cur) => rot1 => ... => rot_i
        future_rot = [cur_rot] * self.future_wind
        future_avel = [self.avel] * self.future_wind
        
        for i in range(1, self.future_wind):
            next_rot, next_avel = SpringUtil.update_spring_rot(
                future_rot[i-1],
                future_avel[i-1],
                desired_rot,
                self.halflife,
                self.dt * self.sub_step
            )
            future_rot[i] = next_rot
            future_avel[i] = next_avel

        future_rot = np.array(future_rot).reshape(-1, 4)
        future_avel = np.array(future_avel).reshape(-1, 3)
        return future_rot, future_avel

    def update(self, task):
        # user input => global desired control info.
        cam_fwd = self.camera_ctrl.cam_fwd
        desired_vel = self._get_target_vel(cam_fwd, self.user_input, self.rotation)
        desired_rot = self._get_target_rot(desired_vel)

        # Predict future trajectory
        future_pos, future_vel, future_acc = self._predict_future_pos(
            self.position, desired_vel
        )
        future_rot, future_avel = self._predict_future_rot(
            self.rotation, desired_rot
        )

        # Update current rotation and position to next frame
        self.next_pos, self.vel, self.acc = SpringUtil.update_spring_pos(
            self.position, self.vel, self.acc, desired_vel, self.halflife, self.dt
        )
        self.next_rot, self.avel = SpringUtil.update_spring_rot(
            self.rotation, self.avel, desired_rot, self.halflife, self.dt
        )
        future_pos[0] = self.next_pos
        future_rot[0] = self.next_rot
        future_vel[0] = self.vel.copy()
        future_avel[0] = self.avel.copy()

        # Update track rendering
        future_rot_wxyz = future_rot[..., [3,0,1,2]]
        for i in range(self.future_wind):
            self.future_nodes[i].set_pos(*future_pos[i])
            self.future_nodes[i].set_quat(p3d.Quat(*future_rot_wxyz[i]))

        # Update track props
        self.future_pos = future_pos
        self.future_rot = future_rot
        self.future_vel = future_vel
        self.future_avel = future_avel
        self.future_acc = future_acc

        return task.cont
    
    def get_desired_state(self):
        return self.future_pos, self.future_rot, self.future_vel, self.future_avel
    