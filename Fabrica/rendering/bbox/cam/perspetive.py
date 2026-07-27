from typing import Optional

from ..utils import Vec3
from .base import BaseCam


class PerspectiveCam(BaseCam):
    def __init__(self,
            origin: Vec3 = (1.0, 1.0, 1.0),
            target: Vec3 = (0.0, 0.0, 0.0),
            up: Vec3 = (0.0, 0.0, 1.0),
            focal_len: Optional[float] = None,
            fov: Optional[float] = None) -> None:
        super().__init__(origin, target, up)
        self.bpy_data.type = 'PERSP'

        if focal_len is None and fov is None:
            focal_len = 50.0
        elif focal_len is not None and fov is not None:
            print('focal_len and fov cannot be set at the same time, selecting focal_len as default.')

        if focal_len is not None:
            self.set_focal_len(focal_len)
        else:
            self.set_fov(fov)
