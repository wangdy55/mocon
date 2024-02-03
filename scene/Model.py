import panda3d.core as p3d
from direct.showbase.DirectObject import DirectObject
import numpy as np

from scene.Scene import Scene

class Model(DirectObject):
    def __init__(self, scene: Scene):
        self.scene = scene

        # Add texture
        end_color = [0, 1, 0, 1]
        body_color = [141/255, 141/255, 170/255, 1]
        self.end_tex = self.add_color_tex(end_color, "end_tex")
        self.body_tex = self.add_color_tex(body_color, "body_tex")

        # Load model npz file
        model_npz = np.load("assets/character/model.npz")
        self.joint_names = model_npz["joint_names"]
        self.parent_idx = model_npz["parent_idx"]

        # Load joints and bodies
        joint_pos = model_npz["joint_pos"]
        body_pos = model_npz["body_pos"]
        body_scale = model_npz["body_scale"]
        self.joints, self.bodies = self.load_model(joint_pos, body_pos, body_scale)
        self.root = self.joints[0] # Define the first joint as root
        
        # a light node to attach light set
        self.light_node = self.scene.render.attach_new_node("light")
        self.light_node.set_pos(
            self.root.get_x(), 0 , self.root.get_z()
        )
        self.load_light()
        self.scene.task_mgr.add(self.update_light, "update_light")

        # joint name index
        self.name2idx = {name: i for i, name in enumerate(self.joint_names)}


    def add_color_tex(self, rgba: list, tex_name: str) -> p3d.Texture:
        # Add a single color texture
        image = p3d.PNMImage(1, 1)
        image.fill(*rgba[:3])
        image.alpha_fill(rgba[3])
        tex = p3d.Texture(tex_name)
        tex.load(image)
        return tex
    
    def load_light(self):
        self.scene.set_background_color((0, 0, 0, 1))
        self.scene.render.set_shader_auto(True)

        ambient_light = p3d.AmbientLight("ambient_light")
        ambient_light.set_color((0.3, 0.3, 0.3, 1))
        self.alnp = self.scene.render.attach_new_node(ambient_light)
        self.scene.render.set_light(self.alnp)
        
        # directional light set
        self.direct_light_set = p3d.NodePath("direct_light_set")

        fill_light1 = p3d.DirectionalLight('fill_light1')
        fill_light1.set_color((0.45, 0.45, 0.45, 1))
        flnp1 = self.direct_light_set.attach_new_node(fill_light1)
        flnp1.set_pos(10, 10, 10)
        flnp1.look_at((0, 0, 0), (0, 1, 0))
        self.scene.render.set_light(flnp1)
        
        fill_light2 = p3d.DirectionalLight("fill_light2")
        fill_light2.set_color((0.45, 0.45, 0.45, 1))
        flnp2 = self.direct_light_set.attach_new_node(fill_light2)
        flnp2.set_pos(-10, 10, 10)
        flnp2.look_at((0, 0, 0), (0, 1, 0))
        self.scene.render.set_light(flnp2)

        key_light = p3d.DirectionalLight("key_light")
        key_light.set_color_temperature(6500)
        key_light.set_shadow_caster(True, 2048, 2048)
        key_light.get_lens().set_film_size((10, 10))
        key_light.get_lens().set_near_far(0.1, 300)
        klnp = self.direct_light_set.attach_new_node(key_light)
        klnp.set_pos(0, 20, -10)
        klnp.look_at((0, 0, 0), (0, 1, 0))
        self.scene.render.set_light(klnp)

        self.direct_light_set.wrt_reparent_to(self.light_node)

    def update_light(self, task) -> int:
        self.light_node.set_pos(
            self.root.get_x(), 0, self.root.get_z()
        )
        return task.cont

    def load_joint(
        self,
        index: int,
        pos: np.ndarray,
        is_end: bool
    ) -> p3d.NodePath:
        # Load a joint with global position
        node = self.scene.render.attach_new_node(f"joint{index}")

        if not is_end:
            cube = self.scene.loader.load_model("assets/character/cube.egg")
            cube.set_texture_off(1)
            cube.set_scale(0.01)
            cube.reparent_to(node)

        node.set_pos(self.scene.render, *pos)
        return node
    
    def load_body(
        self,
        index: int,
        pos: np.ndarray,
        scale: np.ndarray
    ) -> p3d.NodePath:
        # Load a body with global position and size
        cube = self.scene.loader.load_model("assets/character/cube.egg")
        cube.set_texture_off(1)
        cube.set_texture(self.body_tex, 1)
        cube.set_scale(*scale)

        node = self.scene.render.attach_new_node(f"body{index}")
        cube.reparent_to(node)
        node.set_pos(self.scene.render, *pos)
        return node
    
    def load_model(
        self,
        joint_pos: np.ndarray,
        body_pos: np.ndarray,
        body_scale: np.ndarray
    ) -> tuple:
        # Load the model with joints and bodies
        joints, bodies = [], []
        num_joints = len(joint_pos)
        num_bodies = len(body_pos)
        is_end = ["end" in name for name in self.joint_names]

        for i in range(num_joints):
            joints.append(
                self.load_joint(i, joint_pos[i], is_end[i])
            )
            if i >= num_bodies:
                continue
            bodies.append(
                self.load_body(i, body_pos[i], body_scale[i])
            )
            bodies[-1].wrt_reparent_to(joints[i])

        joints = np.array(joints)
        bodies = np.array(bodies)
        return joints, bodies
    
    @property
    def joint_list(self) -> np.ndarray:
        # dtype of joints is panda3d.core.NodePath
        return self.joints
    
    def _set_joint_pos(self, name, pos):
        joint_idx = self.name2idx[name]
        self.joints[joint_idx].set_pos(self.scene.render, *pos)

    def _set_joint_rot(self, name, quat):
        joint_idx = self.name2idx[name]
        self.joints[joint_idx].set_quat(
            self.scene.render,
            p3d.Quat(*quat[..., [3, 0, 1, 2]].tolist())
        )

    def set_joints(self, name, pos, quat):
        self._set_joint_pos(name, pos)
        self._set_joint_rot(name, quat)
        