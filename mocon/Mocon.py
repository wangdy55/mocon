from mocon.character.Character import Character
from mocon.controller.CameraCtrl import CameraCtrl
from mocon.controller.CharacterCtrl import CharacterCtrl
from mocon.motion.MotionCtrl import MotionCtrl
from mocon.motion.SimpleMotionCtrl import SimpleMotionCtrl
from mocon.motion.mocap.BVHPlayer import BVHPlayer

class Mocon:
    def __init__(
            self,
            scene, model,
            random
        ):
        self.scene = scene
        self.model = model

        self.chara = Character(self.scene, self.model, random)
        self.camera_ctrl = CameraCtrl(self.chara)

        if random:
            self.motion_ctrl = MotionCtrl(self.chara)
        else:
            self.chara_ctrl = CharacterCtrl(self.chara, self.camera_ctrl)
            self.motion_ctrl = SimpleMotionCtrl(self.chara, self.chara_ctrl)
