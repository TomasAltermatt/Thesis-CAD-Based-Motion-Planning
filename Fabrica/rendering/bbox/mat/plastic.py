from ..utils import Color
from .base import BaseMat

class PlasticMat(BaseMat):
    def __init__(
            self,
            color: Color = Color.white(),
            use_mesh_color: bool = False,
            ao_strength: float = 0.0,
            ao_distance: float = 10.0,
            roughness: float = 0.3,
            sheen_tint: tuple[float, float, float, float] = (0, 0, 0, 1),
            specular_ior_level: float = 0.5,
            ior: float = 1.45,
            transmission_weight: float = 0.0,
            coat_roughness: float = 0.0) -> None:
        super().__init__()

        nodes = self.nodes
        links = self.links

        bsdf_node = nodes['Principled BSDF']

        bc_node, _ = self.add_color(color, use_mesh_color)
        mix_rgb_node, ao_node, _ = self.add_ao(ao_strength, ao_distance)

        bsdf_node.inputs['Roughness'].default_value = roughness
        bsdf_node.inputs['Sheen Tint'].default_value = sheen_tint
        bsdf_node.inputs['Specular IOR Level'].default_value = specular_ior_level
        bsdf_node.inputs['IOR'].default_value = ior
        bsdf_node.inputs['Transmission Weight'].default_value = transmission_weight
        bsdf_node.inputs['Coat Roughness'].default_value = coat_roughness

        links.new(bc_node.outputs['Color'], ao_node.inputs['Color'])
        links.new(mix_rgb_node.outputs['Color'], bsdf_node.inputs['Base Color'])
