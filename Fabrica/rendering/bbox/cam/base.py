import math
import bpy # pylint: disable=import-error

from ..base import BaseObj
from ..utils import lookat, Vec3

class BaseCam(BaseObj):

    def __init__(self, origin: Vec3, target: Vec3, up: Vec3) -> None:

        bpy.ops.object.camera_add(location=origin, rotation=lookat(origin, target, up))
        super().__init__(bpy.context.object)

    def set_focal_len(self, focal_len: float) -> None:
        self.bpy_data.lens = focal_len
        self.bpy_data.lens_unit = 'MILLIMETERS'

    def set_fov(self, fov: float) -> None:
        self.bpy_data.angle = math.radians(fov)
        self.bpy_data.lens_unit = 'FOV'
