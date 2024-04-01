from mocon.character.Character import Character
from mocon.controller.CameraCtrl import CameraCtrl
from mocon.controller.CharacterCtrl import CharacterCtrl
from mocon.motion.MotionGen import MotionGen
from mocon.motion.MotionCtrl import MotionCtrl

class Mocon:
    def __init__(
            self,
            scene, model,
            mode="control",
        ):
        self.scene = scene
        self.model = model

        self.chara = Character(self.scene, self.model)
        self.camera_ctrl = CameraCtrl(self.chara)

        if mode == "random":
            self.motion_ctrl = MotionGen(self.chara)
        elif mode == "control":
            self.chara_ctrl = CharacterCtrl(self.chara, self.camera_ctrl)
            self.motion_ctrl = MotionCtrl(self.chara, self.chara_ctrl)
