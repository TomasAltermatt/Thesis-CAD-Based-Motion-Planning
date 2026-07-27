import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import pickle
import numpy as np
import trimesh
from planning.robot.util_arm import get_arm_chain, get_gripper_pos_quat_from_arm_q
from planning.robot.geometry import load_part_meshes, load_arm_meshes, load_gripper_meshes, transform_gripper_meshes, transform_arm_meshes, get_buffered_gripper_meshes, get_gripper_meshes_transforms
from planning.robot.workcell import get_assembly_center


def render(gripper_type, gripper_meshes, object_meshes, grasps):
    render_meshes = [mesh for mesh in object_meshes]
    for grasp in grasps:
        gripper_meshes_i = transform_gripper_meshes(gripper_type, gripper_meshes, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio)
        render_meshes += list(gripper_meshes_i.values())
    trimesh.Scene(render_meshes).show()


def visualize_grasps(assembly_dir, log_dir, motion_type, part_id, grasp_id=None):

    asset_folder = os.path.join(project_base_dir, './assets')

    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps_data = pickle.load(fp)
    gripper_type, arm_type = grasps_data['gripper'], grasps_data['arm']
    gripper_meshes = load_gripper_meshes(gripper_type, asset_folder, has_ft_sensor=True)
    # gripper_meshes = get_buffered_gripper_meshes(gripper_type, gripper_meshes)

    part_meshes_final = load_part_meshes(assembly_dir, transform='final')
    part_meshes_final = {part_name.replace('part', ''): part_mesh for part_name, part_mesh in part_meshes_final.items()}
    for part_mesh in part_meshes_final.values():
        part_mesh.apply_translation(get_assembly_center(arm_type))

    grasps = grasps_data['grasps']
    if motion_type == 'move':
        grasps = {grasp[0].grasp_id: grasp[0] for grasp in grasps[part_id][motion_type]}
    elif motion_type == 'hold':
        grasps = {grasp.grasp_id: grasp for grasp in grasps[part_id][motion_type]}
    else:
        raise NotImplementedError
    if grasp_id is None:
        grasps = list(grasps.values())
    else:
        grasps = [grasps[grasp_id]]

    render(gripper_type, gripper_meshes, [part_meshes_final[part_id]], grasps)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='directory of assembly')
    parser.add_argument('--log-dir', type=str, required=True, help='directory to load precedence and save generated grasps')
    parser.add_argument('--motion-type', type=str, default='move', choices=['move', 'hold'])
    parser.add_argument('--part-id', type=str, required=True, help='part')
    parser.add_argument('--grasp-id', type=int, default=None, help='grasp id')
    args = parser.parse_args()

    visualize_grasps(args.assembly_dir, args.log_dir, args.motion_type, args.part_id, args.grasp_id)
