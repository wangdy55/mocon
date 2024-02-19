from mocon.character.Character import Character
from mocon.controller.CameraCtrl import CameraCtrl
from mocon.controller.CharacterCtrl import CharacterCtrl
from mocon.motion.MotionCtrl import MotionCtrl
from mocon.motion.MvaeMotionCtrl import MvaeMotionCtrl

class Mocon:
    def __init__(self, scene, model, use_mvae):
        self.scene = scene
        self.model = model

        self.chara = Character(self.scene, self.model, use_mvae=True)
        self.camera_ctrl = CameraCtrl(self.chara)

        if not use_mvae:
            self.chara_ctrl = CharacterCtrl(self.chara, self.camera_ctrl)
            self.motion_ctrl = MotionCtrl(self.chara, self.chara_ctrl)
        else:
            self.motion_ctrl = MvaeMotionCtrl(self.chara)
