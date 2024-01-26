from scene.Scene import Scene
from scene.Model import Model
from mocon.character.Character import Character
from mocon.controller.CameraController import CameraController
from mocon.controller.CharacterController import CharacterController
from mocon.motion.MotionController import MotionController

def main():
    scene = Scene()
    model = Model(scene)

    character = Character(model)
    cameraController = CameraController(model, scene)
    characterController = CharacterController(cameraController, scene)
    motionController = MotionController(
        character,
        characterController,
        scene,
        bvhPath="mocon/motion/mocap/bvh/walk1_subject5.bvh"
    )
    
    scene.run()

if __name__ == '__main__':
    main()