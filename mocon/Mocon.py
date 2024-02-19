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
            use_mvae,
            play_mod=False
        ):
        self.scene = scene
        self.model = model

        self.chara = Character(self.scene, self.model, use_mvae)
        self.camera_ctrl = CameraCtrl(self.chara)

        if play_mod:
            self.bvh_player = BVHPlayer(
                self.chara,
                bvh_file="mocon/motion/mocap/bvh/run2_subject1.bvh",
                npz_file="mocon/motion/mocap/npz/run2_subject1.npz"
            )
        else:
            if use_mvae:
                self.motion_ctrl = MotionCtrl(self.chara)
            else:
                self.chara_ctrl = CharacterCtrl(self.chara, self.camera_ctrl)
                self.motion_ctrl = SimpleMotionCtrl(self.chara, self.chara_ctrl)
