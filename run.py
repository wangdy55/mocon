from scene.Scene import Scene
from mocon.character.Character import Character
from mocon.controller.CameraController import CameraController
# from mocon.controller.CameraControllerGames import CameraController

def main():
    scene = Scene()
    character = Character(scene.model)
    cameraController = CameraController(scene)
    # CharacterController
    
    scene.run()

if __name__ == '__main__':
    main()