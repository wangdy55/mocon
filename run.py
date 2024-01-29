import os

from scene.Scene import Scene
from scene.Model import Model
from mocon.character.Character import Character
from mocon.controller.CameraController import CameraController
from mocon.controller.CharacterController import CharacterController
from mocon.motion.MotionController import MotionController

bvhPath = "mocon/motion/mocap/bvh/walk1_subject5.bvh"
npzPath = "mocon/motion/mocap/npz/walk1_subject5.npz"
stateDictPath = "mocon/motion/mvae/model/walk1_subject5_state_dict.pt"

def main():
    scene = Scene()
    model = Model(scene)

    character = Character(model, scene, bvhPath)
    cameraController = CameraController(model, scene)
    characterController = CharacterController(cameraController, scene)
    MotionController(
        character, characterController, scene,
        npzPath, stateDictPath
    )
    
    scene.run()

if __name__ == '__main__':
    main()