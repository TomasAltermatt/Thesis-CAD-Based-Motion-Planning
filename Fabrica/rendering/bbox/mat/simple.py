from ..utils import Color
from .base import BaseMat

class SimpleMat(BaseMat):
    def __init__(
            self,
            color: Color = Color.white(),
            use_mesh_color: bool = False) -> None:
        super().__init__()

        nodes = self.nodes
        links = self.links

        bsdf_node = nodes['Principled BSDF']

        bc_node, _ = self.add_color(color, use_mesh_color)

        links.new(bc_node.outputs['Color'], bsdf_node.inputs['Base Color'])
