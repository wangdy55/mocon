import panda3d.core as p3d
from direct.showbase.DirectObject import DirectObject
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.controller.CameraController import CameraController
from mocon.controller.utils.Interpolator import Interpolator
from mocon.controller.utils.Visualizer import Visualizer
from scene.Scene import Scene

from mocon.character.Character import Character

class CharacterController(DirectObject):
    def __init__(
        self,
        chara: Character,
        camera_ctrl: CameraController,
        scene: Scene
    ):
        super().__init__()
        self.chara = chara
        self.camera_ctrl = camera_ctrl
        self.scene = scene
        self.scene.task_mgr.add(self.update, "update_chara_ctrl")
        self.dt = scene.dt

        self.vel = p3d.LVector3(0, 0, 0)
        self.acc = p3d.LVector3(0, 0, 0)
        self.avel = p3d.LVector3(0, 0, 0)
        self.input = p3d.LVector3(0, 0, 0) # user input in camera forward direction

        self.future_wind = 6
        self.future_nodes = []
        self.future_pos = []
        self.future_rot = []
        self.future_vel = []
        self.future_avel = []

        self.target_rot = np.array([0, 0, 0, 1])
        self.target_pos = p3d.LVector3(0, 0, 0)
        self.vel_delta = p3d.LVector3(0, 0, 0)
        self.rot_delta = p3d.LVector3(0, 0, 0)
        self.halflife = 0.27
        self.move_speed = p3d.LVector3(1.75, 1.5, 1.25)

        self.set_key_map()
        # self.init_key_input()

        arrow_color = [0, 1, 0, 1]
        for i in range(self.future_wind):
            node = self.scene.render.attach_new_node(f"future_node{i}")
            node.set_pos(0, 0.01, 0)
            if i == 0:
                Visualizer.draw_arrow(node, color=arrow_color)
            node.reparent_to(self.scene.render)
            self.future_nodes.append(node)
        self._node = self.future_nodes[0]

    @property
    def node(self):
        return self._node
    
    @property
    def rotation(self):
        return np.array(self.node.getQuat())[[1, 2, 3, 0]]
    
    @property
    def position(self):
        return self.node.getPos()
    
    def handle_input(self, axis, val):
        if axis == "x":
            self.input[0] = val
        elif axis == "z":
            self.input[2] = val

    def set_key_map(self):
        key_map = {
            ("w", "arrow_up"): ["z", 1],
            ("s", "arrow_down"): ["z", -1],
            ("a", "arrow_left"): ["x", 1],
            ("d", "arrow_right"): ["x", -1],
        }
        for keys, vals in key_map.items():
            # wasd
            self.scene.accept(keys[0], self.handle_input, vals)
            self.scene.accept(keys[0]+"-up", self.handle_input, [vals[0], 0])
            # arrows
            self.scene.accept(keys[1], self.handle_input, vals)
            self.scene.accept(keys[1]+"-up", self.handle_input, [vals[0], 0])
        
    def init_key_input(self):
        self.cam_ref_node = self.node.attach_new_node("cam_ref_node")
        self.cam_ref_node.set_pos(0, 3, -5)
        self.camera_ctrl.pos = self.cam_ref_node.getPos(self.scene.render)
        self.cam_ref_node.wrt_reparent_to(self.node)

    def get_target_vel(
        self,
        cam_fwd: p3d.LVector3,
        input: p3d.LVector3,
        init_rot: np.ndarray
    ) -> np.ndarray:
        # Get global target velocity to update position
        cam_fwd[1] = 0
        fwd_speed, side_speed, back_speed = self.move_speed
        # rotation around y-axis
        angle = np.arctan2(cam_fwd[0], cam_fwd[2])
        y_rotvec = R.from_rotvec(angle * np.array([0, 1, 0]))
        # coordinate transformation from camera to global
        global_input = y_rotvec.apply(input)

        # coordinate transformation from global to local
        local_target_dir = R.from_quat(init_rot).apply(global_input, inverse=True)
        if local_target_dir[2] > 0:
            local_target_vel = np.array([side_speed, 0, fwd_speed]) * local_target_dir
        else:
            local_target_vel = np.array([side_speed, 0, back_speed]) * local_target_dir

        global_target_vel = R.from_quat(init_rot).apply(local_target_vel)
        return global_target_vel

    def get_target_rot(self, global_target_vel: np.ndarray) -> np.ndarray:
        if np.linalg.norm(global_target_vel) < 1e-5:
            return self.rotation
        else:
            global_target_dir = global_target_vel / np.linalg.norm(global_target_vel)
            y_rotvec = np.arctan2(global_target_dir[0], global_target_dir[2]) * np.array([0, 1, 0])
            return R.from_rotvec(y_rotvec).as_quat()

    def update_pos(self):
        init_pos = self.node.get_pos()
        init_rot = self.rotation
        self.subStep = 20

        # Get global target velocity and rotation
        cam_fwd = self.camera_ctrl.cam_fwd
        cur_target_vel = self.get_target_vel(cam_fwd, self.input, init_rot)
        cur_target_rot = self.get_target_rot(cur_target_vel)
        # self.target_vel = cur_target_vel
        # self.target_rot = cur_target_rot

        self.vel_delta = (cur_target_vel - self.vel) / self.dt
        self.rot_delta = (
            R.from_quat(cur_target_rot).inv() * R.from_quat(init_rot)
        ).as_rotvec() / self.dt

        # Predict future rotations
        new_rot, new_avel = init_rot, self.avel
        rot_trajactory = [new_rot]
        self.future_avel = [new_avel]
        for i in range(self.future_wind):
            new_rot, new_avel = Interpolator.simulation_rotations_update(
                new_rot, new_avel, cur_target_rot, self.halflife, self.dt*self.subStep
            )
            rot_trajactory.append(new_rot)
            self.future_avel.append(new_avel.copy())

        # Predict future positions
        new_pos, new_vel, new_acc = init_pos, self.vel, self.acc
        pos_trajactory = [new_pos]
        self.future_vel = [new_vel]
        for i in range(self.future_wind-1):
            new_pos, new_vel, new_acc = Interpolator.simulation_positions_update(
                new_pos, new_vel, new_acc, cur_target_vel, self.halflife, self.dt*self.subStep
            )
            pos_trajactory.append(new_pos)
            self.future_vel.append(new_vel.copy())

        # Update current rotation and position to next frame
        self.target_rot, self.avel = Interpolator.simulation_rotations_update(
            init_rot, self.avel, cur_target_rot, self.halflife, self.dt
        )
        self.target_pos, self.vel, self.acc = Interpolator.simulation_positions_update(
            init_pos, self.vel, self.acc, cur_target_vel, self.halflife, self.dt
        )
        rot_trajactory[0] = self.target_rot
        rot_trajactory = np.array(rot_trajactory).reshape(-1, 4)
        pos_trajactory[0] = self.target_pos
        pos_trajactory = np.array(pos_trajactory).reshape(-1, 3)
            
        # Record trajectory
        self.future_pos = np.array(pos_trajactory).reshape(-1, 3)
        self.future_rot = rot_trajactory.copy()
        self.future_vel[0] = self.vel.copy()
        self.future_vel = np.array(self.future_vel).reshape(-1, 3)
        self.future_avel[0] = self.avel.copy()
        self.future_avel = np.array(self.future_avel).reshape(-1, 3)

        rot_trajactory = rot_trajactory[..., [3, 0, 1, 2]]
        for i in range(self.future_wind):
            self.future_nodes[i].set_pos(*pos_trajactory[i])
            self.future_nodes[i].set_quat(p3d.Quat(*rot_trajactory[i]))
        
        '''
        # Update camera position to controller
        delta = positionTrajactory[0] - initPos
        delta = p3d.LVector3(*delta)
        self.cameraController.pos += delta
        self.cameraController.center += delta
        self.cameraController.look()
        '''
        # Update camera position to character
        delta = self.chara.root_pos - self.camera_ctrl.center
        delta = p3d.LVector3(*delta)
        self.camera_ctrl.center += delta
        self.camera_ctrl.pos += delta
        self.camera_ctrl.look()

    def draw_future_dir(self):
        pass

    def update(self, task):
        self.update_pos()
        return task.cont
    
    def get_next_state(self):
        return self.target_rot, self.avel, self.target_pos, self.vel