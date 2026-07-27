from pathlib import Path

import numpy as np
import trimesh
from jaxtyping import Float
import bpy # pylint: disable=import-error

from ..utils import PathLike
from .base import BaseShape


class MeshShape(BaseShape):

    def __init__(self, bpy_obj: bpy.types.Object) -> None:
        super().__init__(bpy_obj)
        self.set_location((0, 0, 0))
        self.set_rotation((0, 0, 0))
        self.set_scale((1, 1, 1))

    @classmethod
    def from_file(cls, path: PathLike) -> 'MeshShape':
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'file not found: {path}')

        if path.suffix == '.stl':
            bpy.ops.wm.stl_import(filepath=str(path))
        elif path.suffix == '.obj':
            bpy.ops.wm.obj_import(filepath=str(path))
        elif path.suffix == '.ply':
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            raise ValueError(f'unsupported file: {path}')
        bpy_obj = bpy.context.object
        return cls(bpy_obj)

    @classmethod
    def from_trimesh(cls, tri_mesh: trimesh.Trimesh) -> 'MeshShape':
        bpy_mesh = bpy.data.meshes.new(name='mesh')
        bpy_mesh.from_pydata(tri_mesh.vertices, [], tri_mesh.faces)
        bpy_mesh.update()
        bpy_mesh.validate()
        bpy_obj = bpy.data.objects.new('mesh', bpy_mesh)
        bpy.context.scene.collection.objects.link(bpy_obj)
        return cls(bpy_obj)

    @classmethod
    def from_numpy(cls, vertices: np.ndarray, faces: np.ndarray) -> 'MeshShape':
        bpy_mesh = bpy.data.meshes.new(name='mesh')
        bpy_mesh.from_pydata(vertices, [], faces)
        bpy_mesh.update()
        bpy_mesh.validate()
        bpy_obj = bpy.data.objects.new('mesh', bpy_mesh)
        bpy.context.scene.collection.objects.link(bpy_obj)
        return cls(bpy_obj)

    def set_vertex_colors(self, colors: Float[np.ndarray, "N 3"] | Float[np.ndarray, "N 4"]) -> None:
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.full((colors.shape[0], 1), 1.0, dtype=colors.dtype)], axis=1)

        if colors.shape[0] != self.num_verts:
            raise ValueError('invalid number of vertex colors')

        vertex_color_layer = self.bpy_data.vertex_colors.new(name='Col')

        idx = 0
        for i_face in range(self.num_faces):
            for i_vert in self.bpy_data.polygons[i_face].vertices:
                vertex_color_layer.data[idx].color = colors[i_vert]
                idx += 1

    def set_face_colors(self, colors: Float[np.ndarray, "N 3"] | Float[np.ndarray, "N 4"]) -> None:
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.full((colors.shape[0], 1), 1.0, dtype=colors.dtype)], axis=1)

        if colors.shape[0] != self.num_faces:
            raise ValueError('invalid number of face colors')

        vertex_color_layer = self.bpy_data.vertex_colors.new(name='Col')

        idx = 0
        for i_face in range(self.num_faces):
            for i_vert in self.bpy_data.polygons[i_face].vertices:
                vertex_color_layer.data[idx].color = colors[i_face]
                idx += 1

    def set_full_colors(self, colors: Float[np.ndarray, "N 3 3"] | Float[np.ndarray, "N 3 4"]) -> None:
        if colors.shape[2] == 3:
            colors = np.concatenate([colors, np.full((colors.shape[0], colors.shape[1], 1), 1.0, dtype=colors.dtype)], axis=2)

        if colors.shape[0] != self.num_faces:
            raise ValueError('invalid number of vertex colors')

        vertex_color_layer = self.bpy_data.vertex_colors.new(name='Col')

        for i_face in range(self.num_faces):
            for i_vert in range(3):
                vertex_color_layer.data[i_face * 3 + i_vert].color = colors[i_face, i_vert]
