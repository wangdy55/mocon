from panda3d.core import Vec4, Quat

class ShowUtil:
    @staticmethod
    def draw_circle(nodepath, radius, color):
        from direct.showutil.Rope import Rope
        rope = Rope()
        rope.ropeNode.set_use_vertex_thickness(1)
        rope.ropeNode.set_use_vertex_color(1)

        a = 0.866 * radius
        b = 0.5 * radius
        c = 1 * radius
        w = 0.5
        points = [
            (-a, 0, b,1),
            (0, 0, c, w),
            (a, 0, b, 1),
            (a, 0, -b, w),
            (0, 0, -c, 1),
            (-a, 0, -b, w),
            (-a, 0, b, 1)
        ]

        verts = [
            {
                "node": nodepath,
                "point": point,
                "color": color,
                "thickness": 2,
            } for point in points
        ]

        rope.setup(3, verts, knots=[0, 0, 0, 1/3, 1/3, 2/3, 2/3, 1, 1, 1])
        rope.reparent_to(nodepath)
        return rope

    @staticmethod
    def draw_marker(nodepath, radius, color, circle=False):
        if circle:
            ShowUtil.draw_circle(nodepath, radius, color)

        from direct.showutil import BuildGeometry as BG
        node = nodepath.attach_new_node("arrow")
        BG.addArrowGeom(node, sizeX=0.1, sizeY=0.3, color=color)

        node.set_hpr(0, 90, 180)
        node.set_pos(0, 0, 0.15)
        node.wrt_reparent_to(nodepath)
        return node
