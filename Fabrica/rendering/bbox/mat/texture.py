from typing import Optional
from ..utils import Color, PathLike
from .base import BaseMat

class TextureMat(BaseMat):
    def __init__(
            self,
            texture_path: PathLike,
            colorspace_settings: str = 'sRGB',
            color: Optional[Color] = None,
            roughness: float = 1.0,
            sheen_tint: tuple[float, float, float, float] = (0, 0, 0, 1),
            alpha: float = 1.0) -> None:
        super().__init__()

        nodes = self.nodes
        links = self.links

        bsdf_node = nodes['Principled BSDF']

        if color is None:
            color = Color.black()
        bc_node, hsv_node = self.add_color(color)
        tex_image_node = self.add_texture(texture_path, colorspace_settings)

        bsdf_node.inputs['Roughness'].default_value = roughness
        bsdf_node.inputs['Sheen Tint'].default_value = sheen_tint
        bsdf_node.inputs['Alpha'].default_value = alpha

        links.new(tex_image_node.outputs['Color'], hsv_node.inputs['Color'])
        links.new(hsv_node.outputs['Color'], bc_node.inputs['Color'])
        links.new(bc_node.outputs['Color'], bsdf_node.inputs['Base Color'])
