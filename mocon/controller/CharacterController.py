import panda3d.core as p3d
from direct.showbase.DirectObject import DirectObject
import numpy as np
from scipy.spatial.transform import Rotation as R

from mocon.controller.CameraController import CameraController
from mocon.controller.utils.Interpolator import Interpolator
from mocon.controller.utils.Visualizer import Visualizer
from scene.Scene import Scene

from mocon.character.Character import Character

class CharacterController(DirectObject):
    def __init__(
        self,
        character: Character,
        cameraController: CameraController,
        scene: Scene
    ):
        super().__init__()
        self.character = character
        self.cameraController = cameraController
        self.scene = scene
        self.scene.taskMgr.add(self.update, "updateCharacterController")
        self.dt = scene.dt

        self.vel = p3d.LVector3(0, 0, 0)
        self.acc = p3d.LVector3(0, 0, 0)
        self.avel = p3d.LVector3(0, 0, 0)
        self.camInput = p3d.LVector3(0, 0, 0)

        self.futureWind = 6
        self.futureNodes = []
        self.futurePos = []
        self.futureRot = []
        self.futureVel = []
        self.futureAvel = []

        self.targetRot = np.array([0, 0, 0, 1])
        self.targetPos = p3d.LVector3(0, 0, 0)
        self.targetVelOff = p3d.LVector3(0, 0, 0)
        self.targetRotOff = p3d.LVector3(0, 0, 0)
        self.halflife = 0.27
        self.moveSpeed = p3d.LVector3(1.75, 1.5, 1.25)

        self.setKeyMap()
        # self.initKeyInput()

        arrowColor = [0, 1, 0, 1]
        for i in range(self.futureWind):
            node = self.scene.render.attachNewNode(f"futureNode{i}")
            node.setPos(0, 0.01, 0)
            if i == 0:
                Visualizer.drawArrow(node, color=arrowColor)
            node.reparentTo(self.scene.render)
            self.futureNodes.append(node)
        self._node = self.futureNodes[0]

    @property
    def node(self):
        return self._node
    
    @property
    def rotation(self):
        return np.array(self.node.getQuat())[[1, 2, 3, 0]]
    
    @property
    def position(self):
        return self.node.getPos()
    
    def handleInput(self, axis, val):
        if axis == "x":
            self.camInput[0] = val
        elif axis == "z":
            self.camInput[2] = val

    def setKeyMap(self):
        keyMap = {
            ("w", "arrow_up"): ["z", 1],
            ("s", "arrow_down"): ["z", -1],
            ("a", "arrow_left"): ["x", 1],
            ("d", "arrow_right"): ["x", -1],
        }
        for keys, vals in keyMap.items():
            # wasd
            self.scene.accept(keys[0], self.handleInput, vals)
            self.scene.accept(keys[0]+"-up", self.handleInput, [vals[0], 0])
            # arrows
            self.scene.accept(keys[1], self.handleInput, vals)
            self.scene.accept(keys[1]+"-up", self.handleInput, [vals[0], 0])
        
    def initKeyInput(self):
        self.camRefNode = self.node.attachNewNode("camRefNode")
        self.camRefNode.setPos(0, 3, -5)
        self.cameraController.pos = self.camRefNode.getPos(self.scene.render)
        self.camRefNode.wrtReparentTo(self.node)

    def getTargetVel(
        self,
        camForward: p3d.LVector3,
        camInput: p3d.LVector3,
        initRot: np.ndarray
    ) -> np.ndarray:
        # Get global target velocity to update position
        camForward[1] = 0
        fwdSpeed, sideSpeed, backSpeed = self.moveSpeed
        # rotation around y-axis
        angle = np.arctan2(camForward[0], camForward[2])
        yRotVec = R.from_rotvec(angle * np.array([0, 1, 0]))
        # coordinate transformation from camera to global
        globalInput = yRotVec.apply(camInput)

        # coordinate transformation from global to local
        localTargetDirection = R.from_quat(initRot).apply(globalInput, inverse=True)
        if localTargetDirection[2] > 0:
            localTargetVel = np.array([sideSpeed, 0, fwdSpeed]) * localTargetDirection
        else:
            localTargetVel = np.array([sideSpeed, 0, backSpeed]) * localTargetDirection

        globalTargetVel = R.from_quat(initRot).apply(localTargetVel)
        return globalTargetVel

    def getTargetRot(self, globalTargetVel: np.ndarray) -> np.ndarray:
        if np.linalg.norm(globalTargetVel) < 1e-5:
            return self.rotation
        else:
            targetDirection = globalTargetVel / np.linalg.norm(globalTargetVel)
            yRotVec = np.arctan2(targetDirection[0], targetDirection[2]) * np.array([0, 1, 0])
            return R.from_rotvec(yRotVec).as_quat()

    def updatePos(self):
        initPos = self.node.getPos()
        initRot = self.rotation
        self.subStep = 20

        # Get global target velocity and rotation
        camForward = self.cameraController.camForward
        curTargetVel = self.getTargetVel(camForward, self.camInput, initRot)
        curTargetRot = self.getTargetRot(curTargetVel)
        # self.targetVel = curTargetVel
        # self.targetRot = curTargetRot

        self.targetVelOff = (curTargetVel - self.vel) / self.dt
        self.targetRotOff = (
            R.from_quat(curTargetRot).inv() * R.from_quat(initRot)
        ).as_rotvec() / self.dt

        # Predict future rotations
        newRot, newAvel = initRot, self.avel
        rotationTrajactory = [newRot]
        self.futureAvel = [newAvel]
        for i in range(self.futureWind):
            newRot, newAvel = Interpolator.simulation_rotations_update(
                newRot, newAvel, curTargetRot, self.halflife, self.dt*self.subStep
            )
            rotationTrajactory.append(newRot)
            self.futureAvel.append(newAvel.copy())

        # Predict future positions
        newPos, newVel, newAcc = initPos, self.vel, self.acc
        positionTrajactory = [newPos]
        self.futureVel = [newVel]
        for i in range(self.futureWind-1):
            newPos, newVel, newAcc = Interpolator.simulation_positions_update(
                newPos, newVel, newAcc, curTargetVel, self.halflife, self.dt*self.subStep
            )
            positionTrajactory.append(newPos)
            self.futureVel.append(newVel.copy())

        # Update current rotation and position to next frame
        self.targetRot, self.avel = Interpolator.simulation_rotations_update(
            initRot, self.avel, curTargetRot, self.halflife, self.dt
        )
        self.targetPos, self.vel, self.acc = Interpolator.simulation_positions_update(
            initPos, self.vel, self.acc, curTargetVel, self.halflife, self.dt
        )
        rotationTrajactory[0] = self.targetRot
        rotationTrajactory = np.array(rotationTrajactory).reshape(-1, 4)
        positionTrajactory[0] = self.targetPos
        positionTrajactory = np.array(positionTrajactory).reshape(-1, 3)
            
        # Record trajectory
        self.futurePos = np.array(positionTrajactory).reshape(-1, 3)
        self.futureRot = rotationTrajactory.copy()
        self.futureVel[0] = self.vel.copy()
        self.futureVel = np.array(self.futureVel).reshape(-1, 3)
        self.futureAvel[0] = self.avel.copy()
        self.futureAvel = np.array(self.futureAvel).reshape(-1, 3)

        rotationTrajactory = rotationTrajactory[..., [3, 0, 1, 2]]
        for i in range(self.futureWind):
            self.futureNodes[i].setPos(*positionTrajactory[i])
            self.futureNodes[i].setQuat(p3d.Quat(*rotationTrajactory[i]))

        # Sync. controller to character
        self.node.setX(self.character.rootPos[0])
        self.node.setZ(self.character.rootPos[2])
        
        '''
        # Update camera position to controller
        delta = positionTrajactory[0] - initPos
        delta = p3d.LVector3(*delta)
        self.cameraController.pos += delta
        self.cameraController.center += delta
        self.cameraController.look()
        '''
        # Update camera position to character
        delta = self.character.rootPos - self.cameraController.center
        delta = p3d.LVector3(*delta)
        self.cameraController.center += delta
        self.cameraController.pos += delta
        self.cameraController.look()

    def drawFutureDirections(self):
        pass

    def update(self, task):
        self.updatePos()
        return task.cont
    
    def getNextState(self):
        return self.targetRot, self.avel, self.targetPos, self.vel