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


def render(arm_type, gripper_type, arm_meshes, gripper_meshes, object_meshes, grasps):
    render_meshes = [mesh for mesh in object_meshes]
    for grasp in grasps:
        gripper_meshes_i = transform_gripper_meshes(gripper_type, gripper_meshes, grasp.pos, grasp.quat, np.eye(4), grasp.open_ratio)
        if grasp.arm_pos is not None:
            arm_chain = get_arm_chain(arm_type, base_pos=grasp.arm_pos, base_euler=grasp.arm_euler)
            arm_meshes_i = transform_arm_meshes(arm_meshes, arm_chain, grasp.arm_q)
        else:
            arm_meshes_i = {}
        render_meshes += list(gripper_meshes_i.values()) + list(arm_meshes_i.values())
    trimesh.Scene(render_meshes).show()


def visualize_grasp_pair(assembly_dir, log_dir, part_move, part_hold, grasp_move_id=None, grasp_hold_id=None):
    part_col_manager = trimesh.collision.CollisionManager()
    move_gripper_col_manager = trimesh.collision.CollisionManager()
    hold_gripper_col_manager = trimesh.collision.CollisionManager()

    asset_folder = os.path.join(project_base_dir, './assets')

    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps_data = pickle.load(fp)
    gripper_type, arm_type, has_ft_sensor = grasps_data['gripper'], grasps_data['arm'], grasps_data['ft_sensor']
    arm_meshes = load_arm_meshes(arm_type, asset_folder)
    gripper_meshes = load_gripper_meshes(gripper_type, asset_folder, has_ft_sensor=has_ft_sensor['move'] and has_ft_sensor['hold']) # NOTE: has_ft_sensor is not exactly right
    gripper_meshes = get_buffered_gripper_meshes(gripper_type, gripper_meshes)
    
    part_meshes_final = load_part_meshes(assembly_dir, transform='final')
    for part_name, part_mesh in part_meshes_final.items():
        part_mesh.apply_translation(get_assembly_center(arm_type))
        part_col_manager.add_object(part_name, part_mesh)

    for gripper_name, gripper_mesh in gripper_meshes.items():
        move_gripper_col_manager.add_object(gripper_name + '_move', gripper_mesh)
        hold_gripper_col_manager.add_object(gripper_name + '_hold', gripper_mesh)

    grasps = grasps_data['grasps']
    grasp_id_pairs = grasps_data['grasp_id_pairs'][(part_move, part_hold)]

    grasps_move = {grasp[0].grasp_id: grasp for grasp in grasps[part_move]['move']}
    grasps_hold = {grasp.grasp_id: grasp for grasp in grasps[part_hold]['hold']}

    while True:
        grasp_move_id_curr = np.random.choice(list(grasps_move.keys())) if grasp_move_id is None else grasp_move_id
        grasp_hold_id_curr = np.random.choice(list(grasps_hold.keys())) if grasp_hold_id is None else grasp_hold_id
        # (grasp_move_id_curr, grasp_hold_id_curr) = grasp_id_pairs[np.random.choice(len(grasp_id_pairs))] if grasp_move_id is None or grasp_hold_id is None else (grasp_move_id, grasp_hold_id)
        print('move grasp:', grasp_move_id_curr, 'hold grasp:', grasp_hold_id_curr, 'feasible:', (grasp_move_id_curr, grasp_hold_id_curr) in grasp_id_pairs)

        grasp_move_all, grasp_hold = grasps_move[grasp_move_id_curr], grasps_hold[grasp_hold_id_curr]
        for grasp_move in grasp_move_all:
            gripper_move_transforms = get_gripper_meshes_transforms(gripper_type, gripper_meshes, grasp_move.pos, grasp_move.quat, np.eye(4), grasp_move.open_ratio)
            gripper_hold_transforms = get_gripper_meshes_transforms(gripper_type, gripper_meshes, grasp_hold.pos, grasp_hold.quat, np.eye(4), grasp_hold.open_ratio)
            for gripper_name, gripper_mesh in gripper_meshes.items():
                move_gripper_col_manager.set_transform(gripper_name + '_move', gripper_move_transforms[gripper_name])
                hold_gripper_col_manager.set_transform(gripper_name + '_hold', gripper_hold_transforms[gripper_name])
            
            # Check gripper-part collisions
            for gripper_col_manager in [move_gripper_col_manager, hold_gripper_col_manager]:
                in_collision, collision_data = part_col_manager.in_collision_other(gripper_col_manager, return_data=True)
                if in_collision:
                    max_collision_data = {}
                    for collision in collision_data:
                        collision_names = tuple(collision.names)
                        if collision_names not in max_collision_data:
                            max_collision_data[collision_names] = collision.depth
                        else:
                            max_collision_data[collision_names] = max(max_collision_data[collision_names], collision.depth)
                    print('gripper-part collision:', max_collision_data)
                else:
                    print('no gripper-part collision')
            
            # Check gripper-gripper collisions
            in_collision, collision_data = hold_gripper_col_manager.in_collision_other(move_gripper_col_manager, return_data=True)
            if in_collision:
                max_collision_data = {}
                for collision in collision_data:
                    collision_names = tuple(collision.names)
                    if collision_names not in max_collision_data:
                        max_collision_data[collision_names] = collision.depth
                    else:
                        max_collision_data[collision_names] = max(max_collision_data[collision_names], collision.depth)
                print('gripper-gripper collision:', max_collision_data)
            else:
                print('no gripper-gripper collision')

            render(arm_type, gripper_type, arm_meshes, gripper_meshes, part_meshes_final.values(), [grasp_move, grasp_hold])

        if grasp_move_id is not None or grasp_hold_id is not None or input('Continue? (y/n) ') == 'n':
            break


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='directory of assembly')
    parser.add_argument('--log-dir', type=str, required=True, help='directory to load precedence and save generated grasps')
    parser.add_argument('--part-move', type=str, required=True, help='part to move')
    parser.add_argument('--part-hold', type=str, required=True, help='part to hold')
    parser.add_argument('--grasp-move', type=int, default=None, help='grasp id to move')
    parser.add_argument('--grasp-hold', type=int, default=None, help='grasp id to hold')
    args = parser.parse_args()

    visualize_grasp_pair(args.assembly_dir, args.log_dir, args.part_move, args.part_hold, args.grasp_move, args.grasp_hold)
