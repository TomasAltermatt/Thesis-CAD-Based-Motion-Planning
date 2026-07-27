import math

import numpy as np
import bpy # pylint: disable=import-error

from ..utils import Vec3
from .base import BaseShape


class CylinderShape(BaseShape):

    def __init__(
            self,
            radius: float,
            p1: Vec3 = (0.0, 0.0, 0.0),
            p2: Vec3 = (0.0, 0.0, 0.0)) -> None:

        p1 = np.array(p1)
        p2 = np.array(p2)
        location = (p1 + p2) * 0.5
        depth = np.linalg.norm(p2 - p1)
        rotation = (0.0, math.acos((p2[2] - p1[2]) / depth), math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius, depth=depth, location=location, rotation=rotation)
        bpy_obj = bpy.context.object
        super().__init__(bpy_obj)
