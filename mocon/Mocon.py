from mocon.character.Character import Character
from mocon.controller.CameraCtrl import CameraCtrl
from mocon.controller.CharacterCtrl import CharacterCtrl
from mocon.motion.MotionCtrlRandom import MotionCtrlRandom
from mocon.motion.MotionCtrl import MotionCtrl
from mocon.motion.SimpleMotionCtrl import SimpleMotionCtrl
from mocon.motion.mocap.BVHPlayer import BVHPlayer

class Mocon:
    def __init__(
            self,
            scene, model,
            mode="random",
        ):
        self.scene = scene
        self.model = model

        self.chara = Character(self.scene, self.model)
        self.camera_ctrl = CameraCtrl(self.chara)

        if mode == "random":
            self.motion_ctrl = MotionCtrlRandom(self.chara)
        elif mode == "ctrl":
            self.chara_ctrl = CharacterCtrl(self.chara, self.camera_ctrl)
            self.motion_ctrl = MotionCtrl(self.chara, self.chara_ctrl)
        else:
            self.chara_ctrl = CharacterCtrl(self.chara, self.camera_ctrl)
            self.motion_ctrl = SimpleMotionCtrl(self.chara, self.chara_ctrl)
