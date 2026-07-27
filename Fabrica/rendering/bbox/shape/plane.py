import bpy # pylint: disable=import-error

from ..utils import Vec3
from .base import BaseShape


class PlaneShape(BaseShape):

    def __init__(
            self,
            size: float = 2.0,
            location: Vec3 = (0.0, 0.0, 0.0),
            rotation: Vec3 = (0.0, 0.0, 0.0),
            scale: Vec3 = (1.0, 1.0, 1.0)) -> None:

        bpy.ops.mesh.primitive_plane_add(
            size=size, location=location, rotation=rotation, scale=scale)
        bpy_obj = bpy.context.object
        super().__init__(bpy_obj)
