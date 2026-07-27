from pathlib import Path

import numpy as np
import trimesh
from jaxtyping import Float
import bpy # pylint: disable=import-error

from ..utils import PathLike
from ..mat import BaseMat
from .base import BaseShape


class PointCloudShape(BaseShape):

    def __init__(self, bpy_obj: bpy.types.Object, radius: float) -> None:
        super().__init__(bpy_obj)

        bpy_obj = self.bpy_obj
        bpy_obj.select_set(True)
        bpy.context.view_layer.objects.active = bpy_obj

        bpy.ops.object.modifier_add(type='NODES')
        bpy.ops.node.new_geometry_nodes_modifier()
        tree = bpy_obj.modifiers[-1].node_group

        group_in = tree.nodes['Group Input']
        group_out = tree.nodes['Group Output']
        geometry_node_mesh_to_points = tree.nodes.new('GeometryNodeMeshToPoints')
        geometry_node_mesh_to_points.location.x -= 100
        geometry_node_mesh_to_points.inputs['Radius'].default_value = radius
        self.geometry_node_set_material = tree.nodes.new('GeometryNodeSetMaterial')

        tree.links.new(group_in.outputs['Geometry'], geometry_node_mesh_to_points.inputs['Mesh'])
        tree.links.new(geometry_node_mesh_to_points.outputs['Points'], self.geometry_node_set_material.inputs['Geometry'])
        tree.links.new(self.geometry_node_set_material.outputs['Geometry'], group_out.inputs['Geometry'])

        self.set_normal_mode('vertex')
        self.update()

        self.set_rotation((0, 0, 0))
        self.set_rotation((0, 0, 0))
        self.set_scale((1, 1, 1))

    def set_mat(self, mat: BaseMat) -> None:
        super().set_mat(mat)
        self.geometry_node_set_material.inputs[2].default_value = mat.bpy_data

    @classmethod
    def from_file(cls, path: PathLike, radius: float) -> 'PointCloudShape':
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'file not found: {path}')
        print('Warning: If your point cloud overwrites color, please use other methods.')

        if path.suffix == '.stl':
            bpy.ops.wm.stl_import(filepath=str(path))
        elif path.suffix == '.obj':
            bpy.ops.wm.obj_import(filepath=str(path))
        elif path.suffix == '.ply':
            bpy.ops.wm.ply_import(filepath=str(path))
        else:
            raise ValueError(f'unsupported file: {path}')
        bpy_obj = bpy.context.object
        return cls(bpy_obj, radius)

    @classmethod
    def from_trimesh(cls, tri_mesh: trimesh.PointCloud, radius: float) -> 'PointCloudShape':
        bpy_mesh = bpy.data.meshes.new(name='mesh')
        bpy_mesh.from_pydata(tri_mesh.vertices, [], [])
        bpy_mesh.update()
        bpy_mesh.validate()
        bpy_obj = bpy.data.objects.new('mesh', bpy_mesh)
        bpy.context.scene.collection.objects.link(bpy_obj)
        return cls(bpy_obj, radius)

    @classmethod
    def from_numpy(cls, vertices: np.ndarray, radius: float) -> 'PointCloudShape':
        bpy_mesh = bpy.data.meshes.new(name='mesh')
        bpy_mesh.from_pydata(vertices, [], [])
        bpy_mesh.update()
        bpy_mesh.validate()
        bpy_obj = bpy.data.objects.new('mesh', bpy_mesh)
        bpy.context.scene.collection.objects.link(bpy_obj)
        return cls(bpy_obj, radius)

    def set_colors(self, colors: Float[np.ndarray, "N 3"] | Float[np.ndarray, "N 4"]) -> None:
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.full((colors.shape[0], 1), 1.0, dtype=colors.dtype)], axis=1)

        if colors.shape[0] != self.num_verts:
            raise ValueError('invalid number of vertex colors')

        color_layer = self.bpy_data.attributes.new(name='Col', type='FLOAT_COLOR', domain='POINT')

        for i_color, color in enumerate(colors):
            color_layer.data[i_color].color = color
