from pathlib import Path
import colorsys

import numpy as np
from jaxtyping import Float

import mathutils # pylint: disable=import-error

PathLike = Path | str
Vec3 = tuple[float, float, float] | Float[np.ndarray, "3"]
Mat44 = Float[np.ndarray, "4 4"]


def lookat(
        origin: Vec3 = (1.0, 1.0, 1.0),
        target: Vec3 = (0.0, 0.0, 0.0),
        up: Vec3 = (0.0, 0.0, 1.0)) -> mathutils.Euler:

    origin = np.array(origin)
    target = np.array(target)
    up = np.array(up)

    forward = target - origin
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    rotation_matrix = np.array([right, true_up, -forward]).transpose()
    rotation = mathutils.Matrix(rotation_matrix)
    return rotation.to_euler()


class Color(object):
    def __init__(
            self,
            red: float,
            green: float,
            blue: float,
            alpha: float,
            hue: float,
            saturation: float,
            value: float,
            brightness: float,
            contrast: float) -> None:
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.brightness = brightness
        self.contrast = contrast

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return self.red, self.green, self.blue, self.alpha

    @classmethod
    def from_hsv(
            cls, hue: float, saturation: float, value: float,
            alpha: float = 1.0, brightness: float = 0.0, contrast: float = 0.0) -> 'Color':
        if not all(0.0 <= color <= 1.0 for color in (hue, saturation, value)):
            raise ValueError(f'invalid HSV color: {hue}, {saturation}, {value}')
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        return cls(red, green, blue, alpha, 0.5, 1.0, 1.0, brightness, contrast)

    @classmethod
    def from_rgb(
            cls, red: float | int, green: float | int, blue: float | int,
            alpha: float = 1.0, brightness: float = 0.0, contrast: float = 0.0,
            hue: float = 0.5, saturation: float = 1.0, value: float = 1.0) -> 'Color':
        if all(isinstance(color, int) for color in (red, green, blue)):
            red = red / 255.0
            green = green / 255.0
            blue = blue / 255.0
            print('RGB values are integers, converting to floats')
        if not all(0.0 <= color <= 1.0 for color in (red, green, blue)):
            raise ValueError(f'invalid RGB color: {red}, {green}, {blue}')
        return cls(red, green, blue, alpha, hue, saturation, value, brightness, contrast)

    @classmethod
    def default(cls) -> 'Color':
        return cls.from_rgb(144 / 255, 210 / 255, 236 / 255, 1.0, 0.0, 2.0)

    @classmethod
    def transparent(cls) -> 'Color':
        return cls.from_rgb(0.0, 0.0, 0.0, 0)

    @classmethod
    def white(cls, alpha: float = 1.0, brightness: float = 0.0, contrast: float = 0.0) -> 'Color':
        return cls.from_rgb(1.0, 1.0, 1.0, alpha, brightness, contrast)

    @classmethod
    def black(cls, alpha: float = 1.0, brightness: float = 0.0, contrast: float = 0.0) -> 'Color':
        return cls.from_rgb(0.0, 0.0, 0.0, alpha, brightness, contrast)
