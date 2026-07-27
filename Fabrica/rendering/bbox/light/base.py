import bpy # pylint: disable=import-error

from ..base import BaseObj
from ..utils import Vec3


class BaseLight(BaseObj):
    def __init__(self, type: str, radius: float, location: Vec3, rotation: Vec3, scale: Vec3) -> None:
        bpy.ops.object.light_add(type=type, radius=radius, location=location, rotation=rotation, scale=scale)
        super().__init__(bpy.context.object)
