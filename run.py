from scene.Scene import Scene
from scene.Model import Model
from mocon.character.Character import Character
from mocon.controller.CameraController import CameraController
from mocon.controller.CharacterController import CharacterController
from mocon.controller.MotionController import MotionController

bvh_path = "mocon/motion/mocap/bvh/walk1_subject5.bvh"
npz_path = "mocon/motion/mocap/npz/walk1_subject5.npz"
mvae_path = "mocon/motion/mvae/model/walk1_subject5_240129_190220.pt"

def main():
    scene = Scene()
    model = Model(scene)

    chara = Character(model, scene, bvh_path)
    camera_ctrl = CameraController(chara, scene)
    chara_ctrl = CharacterController(
        chara,
        camera_ctrl,
        scene
    )
    MotionController(
        chara, chara_ctrl, scene,
        npz_path,
        mvae_path
    )
    
    scene.run()

if __name__ == '__main__':
    main()