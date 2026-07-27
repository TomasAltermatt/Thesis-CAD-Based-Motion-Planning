from ..utils import Color
from .base import BaseMat

class DiffuseMat(BaseMat):
    def __init__(self, color: Color = Color.white()) -> None:
        super().__init__()
        self.bpy_data.diffuse_color = color.rgba
