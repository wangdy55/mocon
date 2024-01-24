import panda3d.core as p3d
from direct.showutil import BuildGeometry
import numpy as np

class Visualizer:
    @staticmethod
    def drawArrow(nodePath, color: list):
        node = nodePath.attachNewNode("arrow")
        BuildGeometry.addArrowGeom(node, sizeX=0.1, sizeY=0.3, color=color)
        node.setQuat(p3d.Quat(0, 0, 1, 0) * p3d.Quat(0.707, 0.707, 0, 0))
        node.setPos(0, 0, 0.15)
        node.wrtReparentTo(nodePath)
        return node