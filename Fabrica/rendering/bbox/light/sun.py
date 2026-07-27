from ..utils import lookat, Vec3
from .base import BaseLight


class SunLight(BaseLight):
    def __init__(
            self,
            strength: float,
            angle: float = 0.05,
            origin: Vec3 = (0.0, 0.0, 1.0),
            target: Vec3 = (0.0, 0.0, 0.0)) -> None:
        super().__init__(
            'SUN', 1.0,
            location=(0.0, 0.0, 0.0),
            rotation=lookat(origin, target),
            scale=(1.0, 1.0, 1.0))

        bpy_light = self.bpy_data
        bpy_light.use_nodes = True

        # Set the strength of the light
        bpy_light.node_tree.nodes['Emission'].inputs['Strength'].default_value = strength

        # Angular diameter of the Sun as seen from the Earth (control the softness of the shadow)
        bpy_light.angle = angle
