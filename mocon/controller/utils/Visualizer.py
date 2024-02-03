import panda3d.core as p3d
from direct.showutil import BuildGeometry
import numpy as np

class Visualizer:
    @staticmethod
    def draw_arrow(node_path, color: list):
        node = node_path.attachNewNode("arrow")
        BuildGeometry.addArrowGeom(node, sizeX=0.1, sizeY=0.3, color=color)
        node.set_quat(p3d.Quat(0, 0, 1, 0) * p3d.Quat(0.707, 0.707, 0, 0))
        node.set_pos(0, 0, 0.15)
        node.wrt_reparent_to(node_path)
        return node