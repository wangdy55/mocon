from mocon.character.Character import Character
from mocon.controller.CameraCtrl import CameraCtrl
from mocon.controller.CharacterCtrl import CharacterCtrl
from mocon.motion.mvae.MotionCtrl import MotionCtrl

class Mocon:
    def __init__(self, scene, model):
        self.scene = scene
        self.model = model

        self.chara = Character(self.scene, self.model)
        self.camera_ctrl = CameraCtrl(self.chara)
        self.chara_ctrl = CharacterCtrl(self.chara, self.camera_ctrl)
        self.motion_ctrl = None
