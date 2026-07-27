import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import numpy as np

import bbox

from planning.robot.geometry import load_part_meshes
from rendering.render_utils import SuppressStdout, get_origin_camera, get_color


def get_part_shapes(part_meshes):
    part_colors = get_color(part_meshes.keys())
    part_shapes = {}
    for name, mesh in part_meshes.items():
        mat = bbox.mat.PlasticMat(bbox.utils.Color.from_rgb(*part_colors[name], value=1.5))
        shape = bbox.shape.MeshShape.from_trimesh(mesh)
        shape.set_mat(mat)
        shape.set_normal_mode('vertex', use_auto_smooth=True)
        shape_list = [shape]
        part_shapes[name] = shape_list
    return part_shapes


def render_part(assembly_dir, record_path, verbose=False):

    part_meshes = load_part_meshes(assembly_dir, transform='none', combined=False)

    if verbose:
        print('[render_part] meshes loaded')

    width = 512
    height = 512
    resolution = 100
    spp = 256
    scale = 0.1

    blender = bbox.Blender(width=width, height=height, resolution_percentage=resolution, samples=spp, use_gpu=True, use_denoise=True, max_bounces=4)

    if verbose:
        print('[render_part] blender initialized')

    ground_mat = bbox.mat.InvisibleMat(0.9)
    ground_shape = bbox.shape.PlaneShape(size=1000 * scale, location=(0, 0, 0), rotation=bbox.utils.lookat((0, 0, 0), (0, 0, 1), (1, 0, 0)))
    ground_shape.set_shadow()
    ground_shape.set_mat(ground_mat)
    blender.set_shadow_threshold(0.01)

    sun_light = bbox.light.SunLight(strength=2.0, angle=1.0, origin=(1, -1, 3))

    ambient_color = bbox.utils.Color.from_rgb(0.3, 0.3, 0.3, 1.0)
    blender.set_ambient_light(ambient_color)

    origin = get_origin_camera(horizontal_angle=-60, vertical_angle=55, radius=25)
    target = np.array((0, 0, 0))
    offset = np.array((0, 0, 0))

    origin = (origin + offset) * scale
    target = (target + offset) * scale

    camera = bbox.cam.PerspectiveCam(origin=origin, target=target, up=(0, 0, 1), focal_len=50)

    if verbose:
        print('[render_part] scene initialized')

    all_shapes = get_part_shapes(part_meshes)

    if verbose:
        print('[render_part] shapes initialized')

    shapes = []
    for shape_name, shape_list in all_shapes.items():
        for shape in shape_list:
            shape.set_scale(scale)
            shape.show()
            shapes.append(shape)

    total_mesh = sum([mesh for mesh in part_meshes.values()])
    total_mesh.apply_scale(scale)
    # bounds = total_mesh.bounds.copy()
    # size = np.ptp(bounds, axis=0).max()
    size = 2.0
    for shape in shapes:
        shape.set_scale(scale / size)

    total_mesh.apply_scale(1 / size)
    center = total_mesh.centroid.copy()
    for shape in shapes:
        shape.set_location(-center)

    total_mesh.apply_translation(-center)
    ground_location = (0, 0, total_mesh.bounds[0, 2] - 0.02)
    ground_shape.set_location(ground_location)

    with SuppressStdout():
        blender.render(record_path, cam=camera)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--record-path', type=str, required=True)
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    render_part(args.assembly_dir, args.record_path, args.verbose)
