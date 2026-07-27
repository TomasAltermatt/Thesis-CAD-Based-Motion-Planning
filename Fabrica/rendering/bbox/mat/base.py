from pathlib import Path

import bpy # pylint: disable=import-error

from ..utils import Color, PathLike


class BaseMat(object):
    def __init__(self) -> None:
        self.bpy_data = bpy.data.materials.new(name='MeshMaterial')
        self.bpy_data.use_nodes = True

    @property
    def nodes(self) -> bpy.types.NodeTree:
        return self.bpy_data.node_tree.nodes

    @property
    def links(self) -> bpy.types.NodeLinks:
        return self.bpy_data.node_tree.links

    def add_color(self, color: Color, use_mesh_color: bool = False) -> tuple[bpy.types.Node, ...]:
        nodes = self.nodes
        links = self.links

        hsv_node = nodes.new('ShaderNodeHueSaturation')
        bc_node = nodes.new('ShaderNodeBrightContrast')

        if use_mesh_color:
            attr_node = nodes.new('ShaderNodeAttribute')
            attr_node.attribute_name = 'Col'
            links.new(attr_node.outputs['Color'], hsv_node.inputs['Color'])
        else:
            hsv_node.inputs['Color'].default_value = color.rgba

        hsv_node.inputs['Hue'].default_value = color.hue
        hsv_node.inputs['Saturation'].default_value = color.saturation
        hsv_node.inputs['Value'].default_value = color.value
        hsv_node.location.x -= 200

        bc_node.inputs['Bright'].default_value = color.brightness
        bc_node.inputs['Contrast'].default_value = color.contrast
        bc_node.location.x -= 400

        links.new(hsv_node.outputs['Color'], bc_node.inputs['Color'])

        # links.new(bc_node.outputs['Color'], ao_node.inputs['Color'])
        return bc_node, hsv_node

    def add_ao(self, strength: float = 0.0, distance: float = 10.0) -> tuple[bpy.types.Node, ...]:
        nodes = self.nodes
        links = self.links

        mix_rgb_node = nodes.new('ShaderNodeMixRGB')
        ao_node = nodes.new('ShaderNodeAmbientOcclusion')
        gamma_node = nodes.new('ShaderNodeGamma')

        mix_rgb_node.blend_type = 'MULTIPLY'
        gamma_node.inputs['Gamma'].default_value = strength
        ao_node.inputs['Distance'].default_value = distance
        ao_node.location.x -= 600

        links.new(ao_node.outputs['Color'], mix_rgb_node.inputs['Color1'])
        links.new(ao_node.outputs['AO'], gamma_node.inputs['Color'])
        links.new(gamma_node.outputs['Color'], mix_rgb_node.inputs['Color2'])

        # links.new(mix_rgb_node.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])
        return mix_rgb_node, ao_node, gamma_node

    def add_texture(self, texture_path: PathLike, colorspace_settings: str = 'sRGB') -> bpy.types.Node:
        nodes = self.nodes

        tex_image_node = nodes.new('ShaderNodeTexImage')
        tex_image_node.image = bpy.data.images.load(str(Path(texture_path).resolve()))
        tex_image_node.image.colorspace_settings.name = colorspace_settings
        tex_image_node.location.x -= 800

        # links.new(tex_image_node.outputs['Color'], hsv_node.inputs['Color'])
        return tex_image_node

    def add_edge(self, thickness: float = 0.01, color: Color = Color.black()) -> bpy.types.Node:
        nodes = self.nodes
        links = self.links

        bsdf_node = nodes['Principled BSDF']

        bc_node, _ = self.add_color(color)

        mat_node = nodes.new('ShaderNodeBsdfDiffuse')
        links.new(bc_node.outputs['Color'], mat_node.inputs['Color'])

        edge_node = nodes.new('ShaderNodeWireframe')
        edge_node.inputs[0].default_value = thickness

        mix_node = nodes.new('ShaderNodeMixShader')
        links.new(edge_node.outputs[0], mix_node.inputs[0])
        links.new(bsdf_node.outputs['BSDF'], mix_node.inputs[1])
        links.new(mat_node.outputs['BSDF'], mix_node.inputs[2])
        links.new(mix_node.outputs['Shader'], nodes['Material Output'].inputs['Surface'])
