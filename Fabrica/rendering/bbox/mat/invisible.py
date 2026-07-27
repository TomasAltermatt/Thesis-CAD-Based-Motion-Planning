from .base import BaseMat

class InvisibleMat(BaseMat):
    def __init__(self, transmission_weight: float = 0.0) -> None:
        super().__init__()

        nodes = self.nodes

        bsdf_node = nodes['Principled BSDF']
        bsdf_node.inputs['Transmission Weight'].default_value = transmission_weight
