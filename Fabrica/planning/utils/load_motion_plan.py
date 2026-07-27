import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
import redmax_py as redmax
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import json
import pickle

from utils.renderer import SimRenderer
from assets.load import load_part_ids
from assets.transform import get_transform_matrix, get_pos_euler_from_transform_matrix
from planning.robot.geometry import get_gripper_finger_states, get_gripper_base_name
from planning.robot.util_arm import get_arm_chain, get_gripper_pos_quat_from_arm_q, get_gripper_path_from_arm_path
from planning.robot.sim_string import get_arm_joints
from planning.run_motion_plan import OPEN_RATIO_REST
from rendering.render_motion_plan import create_assembly_dualarm_fixture_xml


def get_camera_option(option):
    if option == 0: # close
        camera_lookat = [-1, 1, 0]
        camera_pos = [1.25, -1.5, 1.5]
    elif option == 1: # far
        camera_lookat = [4.73023, -3.64037, 5.60194]
        camera_pos = [5.34119, -4.31921, 6.00925]
    elif option == 2: # front
        camera_lookat = [0, -4.3295, 3.92641]
        camera_pos = [0, -5.08155, 4.58561]
    else:
        raise NotImplementedError
    return camera_lookat, camera_pos


def roundarr(arr):
    return ['%.3f' % a for a in arr]


def post_process_motion(path):
    """
    Post-process joint angle arrays to avoid unnecessary full rotations.
    """
    processed_path = []
    prev_q_active = None

    for q_active in path:
        q_active = q_active.copy()
        
        if prev_q_active is not None:
            for i in range(len(q_active)):
                # Calculate the difference between the current and previous angle
                diff = q_active[i] - prev_q_active[i]
                
                # If the absolute difference is larger than pi, adjust to the shorter rotation
                if diff > np.pi:
                    q_active[i] -= 2 * np.pi
                elif diff < -np.pi:
                    q_active[i] += 2 * np.pi

        processed_path.append(q_active)
        prev_q_active = q_active  # Update previous angles for the next iteration

    return processed_path


def load_motion_plan(assembly_dir, log_dir):

    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[render_seq_plan] {grasps_path} not found')
        return
    motion_path = os.path.join(log_dir, 'motion.pkl')
    if not os.path.exists(motion_path):
        print(f'[load_motion_plan] {motion_path} not found')
        return
    
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    arm_type, gripper_type, has_ft_sensor = grasps['arm'], grasps['gripper'], grasps['ft_sensor']

    part_ids = load_part_ids(assembly_dir)
    num_parts = len(part_ids)

    with open(motion_path, 'rb') as f:
        motion = pickle.load(f)

    arm_chain_move = get_arm_chain(arm_type, 'move')
    arm_chain_hold = get_arm_chain(arm_type, 'hold')
    arm_chains = {'move': arm_chain_move, 'hold': arm_chain_hold}
    arm_joints = get_arm_joints(arm_type)
    num_dof_arm = len(arm_joints)

    fixture_dir = os.path.join(log_dir, 'fixture')
    assert os.path.exists(fixture_dir)

    xml_string = create_assembly_dualarm_fixture_xml(assembly_dir, fixture_dir, arm_chain_move, arm_chain_hold, gripper_type, has_ft_sensor)
    asset_folder = os.path.join(project_base_dir, 'assets')
    sim = redmax.Simulation(xml_string, asset_folder)

    camera_lookat, camera_pos = get_camera_option(2)
    sim.viewer_options.camera_lookat = camera_lookat
    sim.viewer_options.camera_pos = camera_pos
    
    # set initial joint states
    for arm_joint_name, arm_joint_state in zip(arm_joints, arm_chain_move.rest_q):
        sim.set_joint_q_init(arm_joint_name + '_move', np.array([arm_joint_state]))
    for arm_joint_name, arm_joint_state in zip(arm_joints, arm_chain_hold.rest_q):
        sim.set_joint_q_init(arm_joint_name + '_hold', np.array([arm_joint_state]))
    finger_move_states = get_gripper_finger_states(gripper_type, OPEN_RATIO_REST, suffix='move')
    for finger_name, finger_state in finger_move_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    finger_hold_states = get_gripper_finger_states(gripper_type, OPEN_RATIO_REST, suffix='hold')
    num_dof_finger = len(finger_move_states)
    for finger_name, finger_state in finger_hold_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)

    # degree of freedom mapping
    assert sim.ndof_r == num_parts * 6 + 2 * (6 + num_dof_finger) + 2 * num_dof_arm
    dof_map = {'part': {}, 'move': {'gripper': None, 'arm': None}, 'hold': {'gripper': None, 'arm': {}}}
    dof_map['part'] = {part_ids[i]: i * 6 for i in range(num_parts)}
    dof_map['move']['gripper'] = num_parts * 6
    dof_map['hold']['gripper'] = dof_map['move']['gripper'] + 6 + num_dof_finger
    dof_map['move']['arm'] = dof_map['hold']['gripper'] + 6 + num_dof_finger
    dof_map['hold']['arm'] = dof_map['move']['arm'] + num_dof_arm

    q_curr = sim.get_q()

    for i, motion_step in enumerate(motion):
        q_his = []
        motion_type, body_type, path, active_part, description = motion_step
        print(f'step {i + 1}', motion_type, body_type, active_part, description)

        arm_dof_start = dof_map[motion_type]['arm']
        arm_dof_end = arm_dof_start + num_dof_arm
        gripper_dof_start = dof_map[motion_type]['gripper']
        gripper_dof_end = gripper_dof_start + 6

        if body_type == 'arm':
            
            path = np.vstack([q_curr[arm_dof_start:arm_dof_end], path])
            path_active = post_process_motion(path)[1:]
            path_full = [arm_chains[motion_type].active_to_full(q_active) for q_active in path_active]
            arm_path, gripper_path = path_full, get_gripper_path_from_arm_path(arm_chains[motion_type], path_full, gripper_type, has_ft_sensor=has_ft_sensor[motion_type])
            arm_path_local = path_active
            gripper_base_name = get_gripper_base_name(gripper_type, suffix=motion_type)
            gripper_path_local = [sim.get_joint_q_from_qm(gripper_base_name, gripper_state) for gripper_state in gripper_path]
            
            if active_part is None: # move arm without grasping part
                for arm_state_local, gripper_state_local in zip(arm_path_local, gripper_path_local):
                    q_curr = q_curr.copy()
                    q_curr[arm_dof_start:arm_dof_end] = arm_state_local
                    q_curr[gripper_dof_start:gripper_dof_end] = gripper_state_local
                    q_his.append(q_curr)
            else: # move arm with grasping part
                part_dof_start = dof_map['part'][active_part]
                part_dof_end = part_dof_start + 6
                part_state_curr = q_curr[part_dof_start:part_dof_end]
                gripper_state_curr = q_curr[gripper_dof_start:gripper_dof_end]
                gripper_to_part_transform = np.linalg.inv(get_transform_matrix(sim.get_joint_qm_from_q(gripper_base_name, gripper_state_curr))) @ get_transform_matrix(sim.get_joint_qm_from_q(f'part{active_part}', part_state_curr))
                for arm_state_local, gripper_state_local, gripper_state_global in zip(arm_path_local, gripper_path_local, gripper_path):
                    q_curr = q_curr.copy()
                    q_curr[arm_dof_start:arm_dof_end] = arm_state_local
                    q_curr[gripper_dof_start:gripper_dof_end] = gripper_state_local
                    q_curr[part_dof_start:part_dof_end] = sim.get_joint_q_from_qm(f'part{active_part}', get_pos_euler_from_transform_matrix(get_transform_matrix(gripper_state_global) @ gripper_to_part_transform))
                    q_his.append(q_curr)
        
        elif body_type == 'gripper':
            finger_dof_start = dof_map[motion_type]['gripper'] + 6
            finger_dof_end = finger_dof_start + num_dof_finger
            open_ratio = path
            q_curr = q_curr.copy()
            finger_states = []
            for finger_state in get_gripper_finger_states(gripper_type, open_ratio, suffix=motion_type).values():
                finger_states.extend(finger_state)
            q_curr[finger_dof_start:finger_dof_end] = finger_states
            q_his.append(q_curr)
            continue
        
        if description != 'init':
            sim.set_state_his(q_his, [np.zeros_like(q_his[0]) for _ in range(len(q_his))])
            SimRenderer.replay(sim)

            # if body_type == 'arm':
            #     sim.set_state_his([q_his[-1]], [np.zeros_like(q_his[-1])])
            #     SimRenderer.replay(sim)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    load_motion_plan(args.assembly_dir, args.log_dir)
