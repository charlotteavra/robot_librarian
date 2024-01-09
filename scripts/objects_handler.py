from bodies.body import Body
import numpy as np
from utils import *


class Obj:
    def __init__(
        self,
        object_type: int = None,
        radius: float = 0.0,
        half_extents: tuple = (0, 0, 0),
        length: float = 0.0,
        normal: tuple = (0, 0, 1),
        filename: str = "",
        scale: tuple = (1, 1, 1),
    ):
        self.type = object_type
        self.radius = radius
        self.half_extents = list(half_extents)
        self.length = length
        self.normal = list(normal)
        self.filename = filename
        self.scale = list(scale)


class Sphere(Obj):
    def __init__(self, radius=1):
        super().__init__(object_type=p.GEOM_SPHERE, radius=radius)


class Box(Obj):
    def __init__(self, half_extents=(1, 1, 1)):
        super().__init__(object_type=p.GEOM_BOX, half_extents=half_extents)


class Capsule(Obj):
    def __init__(self, radius=1, length=1):
        super().__init__(object_type=p.GEOM_CAPSULE, radius=radius, length=length)


class Cylinder(Obj):
    def __init__(self, radius=1, length=1):
        super().__init__(object_type=p.GEOM_CYLINDER, radius=radius, length=length)


class Plane(Obj):
    def __init__(self, normal=(0, 0, 1)):
        super().__init__(object_type=p.GEOM_PLANE, normal=normal)


class Mesh(Obj):
    def __init__(self, filename="", scale=(1, 1, 1)):
        super().__init__(object_type=p.GEOM_MESH, filename=filename, scale=scale)


def Shapes(
    env,
    shape,
    static=False,
    mass=1.0,
    positions=((0, 0, 0),),
    orientation=(0, 0, 0, 1),
    visual=True,
    collision=True,
    rgba=(0, 1, 1, 1),
    maximal_coordinates=False,
    return_collision_visual=False,
    position_offset=(0, 0, 0),
    orientation_offset=(0, 0, 0, 1),
):
    collision = (
        env.sim.createCollisionShape(
            shapeType=shape.type,
            radius=shape.radius,
            halfExtents=shape.half_extents,
            height=shape.length,
            fileName=shape.filename,
            meshScale=shape.scale,
            planeNormal=shape.normal,
            collisionFramePosition=position_offset,
            collisionFrameOrientation=orientation_offset,
        )
        if collision
        else -1
    )
    if rgba is not None:
        visual = (
            env.sim.createVisualShape(
                shapeType=shape.type,
                radius=shape.radius,
                halfExtents=shape.half_extents,
                length=shape.length,
                fileName=shape.filename,
                meshScale=shape.scale,
                planeNormal=shape.normal,
                rgbaColor=rgba,
                visualFramePosition=position_offset,
                visualFrameOrientation=orientation_offset,
            )
            if visual
            else -1
        )
    else:
        visual = (
            env.sim.createVisualShape(
                shapeType=shape.type,
                radius=shape.radius,
                halfExtents=shape.half_extents,
                length=shape.length,
                fileName=shape.filename,
                meshScale=shape.scale,
                planeNormal=shape.normal,
                visualFramePosition=position_offset,
                visualFrameOrientation=orientation_offset,
            )
            if visual
            else -1
        )
    if return_collision_visual:
        return collision, visual
    shape_ids = env.sim.createMultiBody(
        baseMass=0 if static else mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=positions[0],
        baseOrientation=get_quaternion(orientation, env),
        batchPositions=positions,
        useMaximalCoordinates=maximal_coordinates,
    )
    shapes = []
    for body in shape_ids:
        shapes.append(Body(body, env, collision_shape=collision, visual_shape=visual))
    return shapes


def Shape(
    env,
    shape,
    static=False,
    mass=1.0,
    position=(0, 0, 0),
    orientation=(0, 0, 0, 1),
    visual=True,
    collision=True,
    rgba=(0, 1, 1, 1),
    maximal_coordinates=False,
    return_collision_visual=False,
    position_offset=(0, 0, 0),
    orientation_offset=(0, 0, 0, 1),
):
    shapes = Shapes(
        env,
        shape,
        static,
        mass,
        positions=(position,),
        orientation=orientation,
        visual=visual,
        collision=collision,
        rgba=rgba,
        maximal_coordinates=maximal_coordinates,
        return_collision_visual=return_collision_visual,
        position_offset=position_offset,
        orientation_offset=orientation_offset,
    )
    return shapes[0]


def URDF(
    env,
    filename,
    static=False,
    position=(0, 0, 0),
    orientation=(0, 0, 0, 1),
    maximal_coordinates=False,
):
    body = env.sim.loadURDF(
        filename,
        basePosition=position,
        baseOrientation=get_quaternion(orientation, env),
        useMaximalCoordinates=maximal_coordinates,
        useFixedBase=static,
    )
    return Body(body, env)


def Ground(env, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
    return URDF(
        filename=os.path.join(assets_directory, "plane", "plane.urdf"),
        env=env,
        static=True,
        position=position,
        orientation=get_quaternion(orientation, env),
    )
    # Randomly set friction of the ground
    # self.ground.set_frictions(self.ground.base, lateral_friction=self.np_random.uniform(0.025, 0.5), spinning_friction=0, rolling_friction=0)


def Line(env, start, end, radius=0.005, rgba=None, rgb=(1, 0, 0), replace_line=None):
    if rgba is None:
        rgba = list(rgb) + [1]
    # line = p.addUserDebugLine(lineFromXYZ=start, lineToXYZ=end, lineColorRGB=rgba[:-1], lineWidth=1, lifeTime=0, physicsClientId=env.id)
    v1 = np.array([0, 0, 1])
    v2 = np.array(end) - start
    orientation = np.cross(v1, v2).tolist() + [
        np.sqrt((np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)) + np.dot(v1, v2)
    ]
    orientation = (
        [0, 0, 0, 1]
        if np.linalg.norm(orientation) == 0
        else orientation / np.linalg.norm(orientation)
    )
    if replace_line is not None:
        replace_line.set_base_pos_orient(
            start + (np.array(end) - start) / 2, orientation
        )
        return replace_line
    else:
        l = Shape(
            env,
            Cylinder(radius=radius, length=np.linalg.norm(np.array(end) - start)),
            static=True,
            position=start + (np.array(end) - start) / 2,
            orientation=orientation,
            collision=False,
            rgba=rgba,
        )
        env.visual_items.append(l)
        return l


def Points(env, point_positions, rgba=(1, 0, 0, 1), radius=0.01, replace_points=None):
    if type(point_positions[0]) not in (list, tuple, np.ndarray):
        point_positions = [point_positions]
    if replace_points is not None:
        for i in range(min(len(point_positions), len(replace_points))):
            replace_points[i].set_base_pos_orient(point_positions[i])
            return replace_points
    else:
        points = Shapes(
            env,
            Sphere(radius=radius),
            static=True,
            positions=point_positions,
            orientation=[0, 0, 0, 1],
            visual=True,
            collision=False,
            rgba=rgba,
        )
        return points


def DebugPoints(
    env, point_positions, points_rgb=((0, 0, 0, 1),), size=10, replace_points=None
):
    if type(point_positions[0]) not in (list, tuple, np.ndarray):
        point_positions = [point_positions]
    if type(points_rgb[0]) not in (list, tuple, np.ndarray):
        points_rgb = [points_rgb] * len(point_positions)
    points = -1
    for i in range(len(point_positions) // 4000 + 1):
        while points < 0:
            if replace_points is None:
                points = env.sim.addUserDebugPoints(
                    pointPositions=point_positions[i * 4000 : (i + 1) * 4000],
                    pointColorsRGB=points_rgb[i * 4000 : (i + 1) * 4000],
                    pointSize=size,
                    lifeTime=0,
                )
            else:
                points = env.sim.addUserDebugPoints(
                    pointPositions=point_positions[i * 4000 : (i + 1) * 4000],
                    pointColorsRGB=points_rgb[i * 4000 : (i + 1) * 4000],
                    pointSize=size,
                    lifeTime=0,
                    replaceItemUniqueId=replace_points,
                )
    return points
