from ..utils import Color
from .base import BaseMat

class TransparentMat(BaseMat):
    def __init__(
            self,
            color: Color = Color.white(),
            alpha: float = 0.5,
            transmission: float = 0.5,
            roughness: float = 0.7,
            sheen_tint: tuple[float, float, float, float] = (0, 0, 0, 1),
            metallic: float = 0.0) -> None:
        super().__init__()

        nodes = self.nodes
        links = self.links

        bsdf_node = nodes['Principled BSDF']

        bc_node, _ = self.add_color(color)

        bsdf_node.inputs['Alpha'].default_value = alpha
        bsdf_node.inputs['Transmission Weight'].default_value = transmission
        bsdf_node.inputs['Roughness'].default_value = roughness
        bsdf_node.inputs['Sheen Tint'].default_value = sheen_tint
        bsdf_node.inputs['Metallic'].default_value = metallic


        links.new(bc_node.outputs['Color'], bsdf_node.inputs['Base Color'])
