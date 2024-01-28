from mocon.character.Character import Character
from mocon.controller.CharacterController import CharacterController
from mocon.motion.utils.BVHMotion import BVHLoader
from scene.Scene import Scene

class MotionController:
    def __init__(
        self,
        character: Character,
        characterController: CharacterController,
        scene: Scene,
        bvhPath: str
    ):
        self.character = character
        self.characterController = characterController
        self.scene = scene
        self.scene.taskMgr.add(self.update, "updateMotion")
        self.bvhLoader = BVHLoader(bvhPath)
        self.jointTrans, self.jointOrien = self.bvhLoader.batchForwardKinematics()
        self.curFrame = 0

    def update(self, task):
        self.character.updateJoints(
            jointNames=self.bvhLoader.jointNames,
            jointTrans=self.jointTrans[self.curFrame],
            jointOrien= self.jointOrien[self.curFrame]
        )
        self.curFrame = (self.curFrame + 1) % (self.bvhLoader.numFrames)
        return task.cont
