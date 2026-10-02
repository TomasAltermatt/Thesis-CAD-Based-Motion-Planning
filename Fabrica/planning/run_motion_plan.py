import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)


import numpy as np
import random
import json
import pickle
from tqdm import tqdm
import traceback
import trimesh
from scipy.spatial.transform import Rotation as R
from time import time

from planning.robot.geometry import load_part_meshes
from planning.robot.motion_plan_arm import ArmMotionPlanner
from planning.robot.util_arm import get_arm_chain, get_ik_target_orientation, get_ft_pos_from_gripper_pos_quat
from planning.robot.workcell import get_dual_arm_box, get_assembly_center
from assets.load import load_pos_quat_dict
from assets.transform import get_transform_matrix_quat, get_transform_matrix, mat_to_pos_quat
from utils.common import TimeStamp
from planning.run_seq_plan import SequencePlanner
from planning.run_seq_opt import SequenceOptimizer
from planning.config import RETRACT_OPEN_RATIO, OPEN_RATIO_REST, RETRACT_DELTA_FAR


def get_pickup_gripper_pose(grasp, pickup_pose, final_pose, lift=0):
    gripper_final_pose = get_transform_matrix_quat(grasp.pos, grasp.quat)
    gripper_pickup_pose = pickup_pose @ np.linalg.inv(final_pose) @ gripper_final_pose
    gripper_pickup_pose[2, 3] += lift
    return gripper_pickup_pose


def get_pickup_arm_q(motion_planner, grasp, pickup_pose, final_pose, arm_q_init=None, lift=0, has_ft_sensor=False, optimizer=None, regularization=None):
    if arm_q_init is None: arm_q_init = grasp.arm_q
    gripper_pickup_pose = get_pickup_gripper_pose(grasp, pickup_pose, final_pose, lift)
    gripper_pickup_pos, gripper_pickup_quat = mat_to_pos_quat(gripper_pickup_pose)
    gripper_pickup_ori = get_ik_target_orientation(motion_planner.arm_chain, motion_planner.gripper_type, gripper_pickup_quat)
    if has_ft_sensor:
        gripper_pickup_pos = get_ft_pos_from_gripper_pos_quat(motion_planner.gripper_type, gripper_pickup_pos, gripper_pickup_quat)
    arm_q_pickup = motion_planner.inverse_kinematics(gripper_pickup_pos, gripper_pickup_ori, q_init=arm_q_init, optimizer=optimizer, regularization_parameter=regularization)
    return arm_q_pickup


def get_back_retract_dir(motion_planner, arm_q):
    fk = motion_planner.arm_chain.forward_kinematics(arm_q)
    if motion_planner.arm_type == 'ur5e':
        retract_dir = -fk[:3, 0]
    else:
        retract_dir = -fk[:3, 2]
    retract_dir /= np.linalg.norm(retract_dir)
    return retract_dir


def get_disassembly_retract_dir(motion_planner, arm_q_assembled, arm_q_disassembled):
    fk_assembled = motion_planner.arm_chain.forward_kinematics(arm_q_assembled)
    fk_disassembled = motion_planner.arm_chain.forward_kinematics(arm_q_disassembled)
    retract_dir = fk_disassembled[:3, 3] - fk_assembled[:3, 3]
    retract_dir /= np.linalg.norm(retract_dir)
    return retract_dir


def get_transformed_part_meshes(part_meshes, part_poses):
    transformed_part_meshes = {}
    for part_id, part_mesh in part_meshes.items():
        part_pose = part_poses[part_id]
        transformed_part_meshes[part_id] = part_mesh.copy()
        transformed_part_meshes[part_id].apply_transform(part_pose)
    return transformed_part_meshes


def post_process_q(arm_chain, q, last_q=None):
    """
    Post-process joint angles to avoid unnecessary full round rotations.
    """
    if last_q is None: return np.array(q)

    q_active = arm_chain.active_from_full(q)
    last_q_active = arm_chain.active_from_full(last_q)
    bounds = arm_chain.get_active_link_bounds()
    assert len(q_active) == len(last_q_active) == len(bounds)

    for i in range(len(q_active)):
        
        # Calculate the difference between the current and previous angle
        diff = q_active[i] - last_q_active[i]
        
        # If the absolute difference is larger than pi, adjust to the shorter rotation
        if diff > np.pi:
            if q_active[i] - 2 * np.pi >= bounds[i][0]:
                q_active[i] -= 2 * np.pi
        elif diff < -np.pi:
            if q_active[i] + 2 * np.pi <= bounds[i][1]:
                q_active[i] += 2 * np.pi

        assert q_active[i] >= bounds[i][0] and q_active[i] <= bounds[i][1], f'{q_active[i]} not in {bounds[i]}'
        assert last_q_active[i] >= bounds[i][0] and last_q_active[i] <= bounds[i][1], f'{last_q_active[i]} not in {bounds[i]}'
    
    return arm_chain.active_to_full(q_active)


def post_process_motion(arm_chain, path, last_q=None):
    """
    Post-process joint angle arrays to avoid unnecessary full round rotations.
    """
    processed_path = []

    for q in path:
        if last_q is not None:
            q = post_process_q(arm_chain, q, last_q)
        processed_path.append(q)
        last_q = q  # Update previous angles for the next iteration

    return processed_path

# this is only for rendering purposes (hardcoded arm sides)
def side_to_channel(side):
    # 'right' maps to Redmax's '_move' channel
    # 'left' maps to Redmax's '_hold' channel
    return 'move' if side == 'right' else 'hold'

def run_motion_plan(assembly_dir, log_dir, optimized, seed, verbose=False):

    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    if not os.path.exists(precedence_path):
        print(f'[run_motion_plan] {precedence_path} not found')
        return
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not os.path.exists(grasps_path):
        print(f'[run_motion_plan] {grasps_path} not found')
        return
    fixture_dir = os.path.join(log_dir, 'fixture')
    pickup_path = os.path.join(fixture_dir, 'pickup.json')
    if not os.path.exists(pickup_path):
        print(f'[run_motion_plan] {pickup_path} not found')
        return
    tree_path = os.path.join(log_dir, 'tree_opt.pkl') if optimized else os.path.join(log_dir, 'tree.pkl')
    if not os.path.exists(tree_path):
        print(f'[run_motion_plan] {tree_path} not found')
        return

    with open(precedence_path, 'rb') as fp:
        G_preced = pickle.load(fp)
    with open(grasps_path, 'rb') as fp:
        grasps = pickle.load(fp)
    gripper_type, arm_type, has_ft_sensor = grasps['gripper'], grasps['arm'], grasps['ft_sensor']

    asset_folder = os.path.join(project_base_dir, './assets')
    part_pos_dict_final, part_quat_dict_final = load_pos_quat_dict(assembly_dir, transform='final')
    part_pos_dict_final = {k: v + get_assembly_center(arm_type) for k, v in part_pos_dict_final.items()}
    part_final_pose = {part_id: get_transform_matrix_quat(part_pos_dict_final[part_id], part_quat_dict_final[part_id]) for part_id in part_pos_dict_final.keys()}
    part_meshes = load_part_meshes(assembly_dir, transform='none')
    part_meshes = {k[4:]: v for k, v in part_meshes.items()} # remove part prefix
    part_ids = list(part_meshes.keys())
  
    with open(pickup_path, 'r') as f:
        part_pickup_pose = json.load(f)
    part_pickup_pose = {part_id: get_transform_matrix(part_pickup_state) for part_id, part_pickup_state in part_pickup_pose.items()}
    fixture_mesh = trimesh.load_mesh(os.path.join(fixture_dir, 'fixture.obj'))
    fixture_height = fixture_mesh.bounds[1, 2] - fixture_mesh.bounds[0, 2]

    part_meshes_pickup = get_transformed_part_meshes(part_meshes, part_pickup_pose)
    part_meshes_final = get_transformed_part_meshes(part_meshes, part_final_pose)
    part_meshes_map = {'pickup': part_meshes_pickup, 'final': part_meshes_final}
    gripper_pickup_pose = {}

    with open(tree_path, 'rb') as fp:
        tree = pickle.load(fp)
    if optimized:
        seq_optimizer = SequenceOptimizer(G_preced, grasps)
        sequence, grasps_sequence = seq_optimizer.get_sequence(tree)
    else:
        seq_planner = SequencePlanner(asset_folder, assembly_dir, G_preced, grasps, save_sdf=True, contact_eps=None)
        sequence, grasps_sequence = seq_planner.sample_sequence(tree, seed=seed)

    if sequence is None or grasps_sequence is None:
        print(f'[run_motion_plan] No feasible sequence found in {tree_path}')
        return
    
    sequence, grasps_sequence = sequence[::-1], grasps_sequence[::-1] # reverse the sequence to be forward assembly

    # --- FIX 1: UNIVERSAL PHYSICAL PLANNERS ---
    physical_chains = {
        'right': get_arm_chain(arm_type, 'move', side='right'), 
        'left': get_arm_chain(arm_type, 'hold', side='left')
    }
    b_m, b_h = get_dual_arm_box(arm_type)
    stamp = TimeStamp()
    random.seed(seed); np.random.seed(seed)
    physical_planners = {
        'right': ArmMotionPlanner(physical_chains['right'], gripper_type, has_ft_sensor['move'], b_m, stamp=stamp),
        'left': ArmMotionPlanner(physical_chains['left'], gripper_type, has_ft_sensor['hold'], b_h, stamp=stamp)
    }
    arm_chains = {'move': physical_chains['right'], 'hold': physical_chains['left']}
    
    rest_q_move = arm_chains['move'].active_to_full(arm_chains['move'].rest_q)
    rest_q_hold = arm_chains['hold'].active_to_full(arm_chains['hold'].rest_q)
    # ------------------------------------------

    '''
    Plan commands
    ['move/hold', 'arm/gripper', q/open ratio, active_part, task]
    Tasks:
    - gripper: init, open, close
    - arm: init, transport, switch, assembly
    '''

    max_speed = { # speed in cm/s
        'transport': 4,
        'switch': 4,
        'assembly': 0.3,
    }
    
    # init
    commands = []
    curr_move_side = 'right'
    curr_hold_side = 'left'

    # appended current arm and gripper states
    commands.append(['move', 'arm', rest_q_move, None, 'init', curr_move_side])
    commands.append(['hold', 'arm', rest_q_hold, None, 'init', curr_hold_side])
    commands.append(['move', 'gripper', OPEN_RATIO_REST, None, 'init', curr_move_side])
    commands.append(['hold', 'gripper', OPEN_RATIO_REST, None, 'init', curr_hold_side])

    for step, ((part_move, part_hold), (grasps_move, grasp_hold)) in enumerate(zip(sequence, grasps_sequence)):
        start_cmd_idx = len(commands)

        # This is to get the side that moves/holds
        curr_move_side = grasps_move[0].grasp_id.split('_')[-1]
        curr_hold_side = grasp_hold.grasp_id.split('_')[-1]
        chan_move = side_to_channel(curr_move_side)
        chan_hold = side_to_channel(curr_hold_side)

        motion_planner_move = physical_planners[curr_move_side]
        motion_planner_hold = physical_planners[curr_hold_side]
        arm_chains['move'] = physical_chains[curr_move_side]
        arm_chains['hold'] = physical_chains[curr_hold_side]
        rest_q_move = arm_chains['move'].active_to_full(arm_chains['move'].rest_q)
        rest_q_hold = arm_chains['hold'].active_to_full(arm_chains['hold'].rest_q)
    
        # hold (first, and pre-move)
        if step == 0:
            random.seed(seed); np.random.seed(seed)
            pickup_q_hold = get_pickup_arm_q(motion_planner_hold, grasp_hold, part_pickup_pose[part_hold], part_final_pose[part_hold], arm_q_init=rest_q_hold, has_ft_sensor=has_ft_sensor['hold'], optimizer='least_squares', regularization=1.0)
            if pickup_q_hold is None:
                raise Exception(f'[run_motion_plan] Failed to solve pickup IK for hold arm in step {step} ({assembly_dir})')
            gripper_pickup_pose[part_hold] = get_pickup_gripper_pose(grasp_hold, part_pickup_pose[part_hold], part_final_pose[part_hold])
            open_ratio_retract_hold = min(grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
            commands.append([chan_hold, 'gripper', open_ratio_retract_hold, None, 'open', curr_hold_side])
            commands.append([chan_hold, 'arm', (pickup_q_hold, [None, np.array([0, 0, 1.0])]), None, 'transport', curr_hold_side])
            commands.append([chan_hold, 'gripper', grasp_hold.open_ratio, None, 'close', curr_hold_side])
            commands.append([chan_hold, 'arm', (grasp_hold.arm_q, [np.array([0, 0, 1.0]), np.array([0, 0, 1.0])]), part_hold, 'transport', curr_hold_side])
        else:
            prev_grasp_hold = grasps_sequence[step - 1][1]
            prev_hold_side = prev_grasp_hold.grasp_id.split('_')[-1]
            chan_prev_hold = side_to_channel(prev_hold_side)

            # if we're changing hold sides, we use the handoff before assembling the next part
            if curr_hold_side != prev_hold_side:
                # handoff logic

                # close gripper for new hold side
                open_ratio_retract_hold = min(grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
                commands.append([chan_hold, 'arm', (grasp_hold.arm_q, [None, None], open_ratio_retract_hold), None, 'switch', curr_hold_side])
                commands.append([chan_hold, 'gripper', grasp_hold.open_ratio, None, 'close', curr_hold_side])

                # open gripper on previous hold side
                open_ratio_retract_prev_hold = min(prev_grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
                commands.append([chan_prev_hold, 'gripper', open_ratio_retract_prev_hold, None, 'open', prev_hold_side])
                #commands.append([chan_prev_hold, 'arm', (rest_q_hold, [None, None]), None, 'transport', prev_hold_side])
                # may need to change this rest_q_hold so it doesnt swing much

            # if we do not change hold sides, it is an arm regrasp (just in case)    
            elif grasp_hold.grasp_id != prev_grasp_hold.grasp_id:
                open_ratio_retract_hold = min(grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
                commands.append([chan_hold, 'gripper', open_ratio_retract_hold, None, 'open', curr_hold_side])
                commands.append([chan_hold, 'arm', (grasp_hold.arm_q, [None, None], open_ratio_retract_hold), None, 'switch', curr_hold_side])
                commands.append([chan_hold, 'gripper', grasp_hold.open_ratio, None, 'close', curr_hold_side])
            
        # move (assembly)
        random.seed(seed); np.random.seed(seed)
        pickup_q_move = get_pickup_arm_q(motion_planner_move, grasps_move[0], part_pickup_pose[part_move], part_final_pose[part_move], arm_q_init=rest_q_move, has_ft_sensor=has_ft_sensor['move'], optimizer='least_squares', regularization=1.0)
        if pickup_q_move is None:
            raise Exception(f'[run_motion_plan] Failed to solve pickup IK for move arm in step {step} ({assembly_dir})')
        gripper_pickup_pose[part_move] = get_pickup_gripper_pose(grasps_move[0], part_pickup_pose[part_move], part_final_pose[part_move])
        open_ratio_retract_move = min(grasps_move[0].open_ratio + RETRACT_OPEN_RATIO, 1.0)
        commands.append([chan_move, 'arm', (pickup_q_move, [None, None], open_ratio_retract_move), None, 'switch', curr_move_side]) 
        commands.append([chan_move, 'gripper', grasps_move[0].open_ratio, None, 'close', curr_move_side])
        commands.append([chan_move, 'arm', (grasps_move[-1].arm_q, [np.array([0, 0, 1.0]), get_disassembly_retract_dir(motion_planner_move, grasps_move[0].arm_q, grasps_move[-1].arm_q)]), part_move, 'transport', curr_move_side])
        commands.append([chan_move, 'arm', grasps_move[0].arm_q, part_move, 'assembly', curr_move_side])

        # open gripper of moving arm after inserting part
        commands.append([chan_move, 'gripper', open_ratio_retract_move, None, 'open', curr_move_side])

        if step == len(sequence) - 1:
            # --- FIX: CLEAR THE WORKSPACE FIRST ---
            # 1. Retract the move arm (right) to its rest pose to get it out of the way
            commands.append([chan_move, 'arm', (rest_q_move, [None, None]), None, 'transport', curr_move_side]) 
            commands.append([chan_move, 'gripper', OPEN_RATIO_REST, None, 'close', curr_move_side])

            # 2. Safely retract the hold arm (left) now that the air is clear
            open_ratio_retract_hold = min(grasp_hold.open_ratio + RETRACT_OPEN_RATIO, 1.0)
            commands.append([chan_hold, 'gripper', open_ratio_retract_hold, None, 'open', curr_hold_side])
            commands.append([chan_hold, 'arm', (rest_q_hold, [None, None]), None, 'transport', curr_hold_side])
            commands.append([chan_hold, 'gripper', OPEN_RATIO_REST, None, 'close', curr_hold_side])
            # --------------------------------------
        


    # post-process qs in commands
    last_qs = {'right': None, 'left': None}
    for command in commands:
        motion_type, body_type, value = command[:3]
        physical_side = command[5]
        if body_type == 'arm':
            if type(value) == tuple:
                q = value[0]
            else:
                q = value
            q = post_process_q(physical_chains[physical_side], q, last_qs[physical_side])
            last_qs[physical_side] = q
            if type(value) == tuple:
                command[2] = (q, *value[1:])
            else:
                command[2] = q

    '''
    Plan path
    ['move/hold', 'arm/gripper', path, active_part, task]
    '''

    paths = []
    current_states = {
        'parts': {part_id: 'pickup' for part_id in part_ids},
        'right': {'arm': rest_q_move, 'gripper': OPEN_RATIO_REST},
        'left': {'arm': rest_q_hold, 'gripper': OPEN_RATIO_REST},
    }
    has_ft_sensor_dict = {'right': has_ft_sensor['move'], 'left': has_ft_sensor['hold']}

    for cid, command in enumerate(commands):
        motion_type, body_type, value, active_part, task, physical_side = command[:6]
        
        if verbose:
            print('[run_motion_plan] side: %s, body: %s, part: %s, task: %s' % (physical_side, body_type, active_part, task))

        assert motion_type in ['move', 'hold'] and body_type in ['arm', 'gripper']
        assert task in ['init', 'open', 'close', 'transport', 'switch', 'assembly']
            
        if task in ['init', 'open', 'close']:
            if task == 'init':
                path = [np.array(value)] if body_type == 'arm' else value
            else:
                path = value
            paths.append([motion_type, body_type, path, active_part, task, physical_side])
            current_states[physical_side][body_type] = value

        elif task in ['transport', 'switch', 'assembly']:
            motion_planner = physical_planners[physical_side]
            q_start = current_states[physical_side]['arm']
            open_ratio = current_states[physical_side]['gripper']

            part_meshes_curr = {part_id: part_meshes_map[current_states['parts'][part_id]][part_id] for part_id in part_ids}
            other_side = 'left' if physical_side == 'right' else 'right'
            
            arm_chain_other = physical_chains[other_side]
            arm_q_other = current_states[other_side]['arm']
            open_ratio_other = current_states[other_side]['gripper']
            has_ft_other = has_ft_sensor_dict[other_side]

            if task == 'transport':
                q_goal, [retract_start, retract_goal] = value
                if retract_start is None: retract_start = get_back_retract_dir(motion_planner, q_start)
                if retract_goal is None: retract_goal = get_back_retract_dir(motion_planner, q_goal)

                if active_part is not None: # transport with part (pickup -> assembly)
                    part_meshes_rest = [v for k, v in part_meshes_curr.items() if k != active_part]
                    path = motion_planner.plan_path_with_grasp(q_start, q_goal,
                        move_pickup_mesh=part_meshes_pickup[active_part], gripper_pickup_transform=gripper_pickup_pose[active_part], 
                        still_meshes=part_meshes_rest + [fixture_mesh], open_ratio=open_ratio, 
                        arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_other,
                        retract_start=retract_start, retract_goal=retract_goal, retract_delta=RETRACT_DELTA_FAR,
                        max_speed=max_speed[task], verbose=verbose)
                    current_states['parts'][active_part] = 'final'
                else: # transport without part
                    path = motion_planner.plan_path(q_start, q_goal,
                        part_meshes=list(part_meshes_curr.values()) + [fixture_mesh], open_ratio=open_ratio, 
                        arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_other,
                        retract_start=retract_start, retract_goal=retract_goal, retract_delta=RETRACT_DELTA_FAR,
                        max_speed=max_speed[task], verbose=verbose)
                
                if path is None:
                    raise Exception(f'[run_motion_plan] Failed to plan path for {physical_side} {body_type} in task {task} | Active Part: {active_part} ({assembly_dir})')
                paths.append([motion_type, body_type, path, active_part, task, physical_side])
                current_states[physical_side][body_type] = q_goal

            elif task == 'switch':
                assert active_part is None
                q_goal, [retract_start, retract_goal], open_ratio_next = value
                if retract_start is None: retract_start = get_back_retract_dir(motion_planner, q_start)
                if retract_goal is None: retract_goal = get_back_retract_dir(motion_planner, q_goal)

                # TODO: update
                path1, path2 = motion_planner.plan_path_switch(q_start, q_goal,
                        part_meshes=list(part_meshes_curr.values()) + [fixture_mesh], open_ratio=open_ratio, open_ratio_next=open_ratio_next,
                        arm_chain_other=arm_chain_other, arm_q_other=arm_q_other, open_ratio_other=open_ratio_other, has_ft_sensor_other=has_ft_other,
                        retract_start=retract_start, retract_goal=retract_goal, retract_delta=RETRACT_DELTA_FAR,
                        max_speed=max_speed[task], verbose=verbose)
                if None in [path1, path2]:
                    raise Exception(f'[run_motion_plan] Failed to plan path for {physical_side} {body_type} in task {task} | Active Part: {active_part} ({assembly_dir})')
                if open_ratio == open_ratio_next:
                    paths.append([motion_type, 'arm', path1 + path2, None, task, physical_side])
                else:
                    if len(path1) > 0:
                        paths.append([motion_type, 'arm', path1, None, task, physical_side])
                    paths.append([motion_type, 'gripper', open_ratio_next, None, 'open', physical_side])
                    if len(path2) > 0:
                        paths.append([motion_type, 'arm', path2, None, task, physical_side])
                current_states[physical_side]['arm'] = q_goal
                current_states[physical_side]['gripper'] = open_ratio_next

            elif task == 'assembly':
                q_goal = value
                path = motion_planner.plan_path_straight(q_start, q_goal, open_ratio, max_speed=max_speed[task], sanity_check=task != 'assembly', verbose=verbose) 
            
                if path is None:
                    raise Exception(f'[run_motion_plan] Failed to plan path for {physical_side} {body_type} in task {task} ({assembly_dir})')
                paths.append([motion_type, body_type, path, active_part, task, physical_side])
                current_states[physical_side][body_type] = q_goal

            else:
                raise NotImplementedError

    # post-process motions in paths
    last_qs = {'right': None, 'left': None}
    for i in range(len(paths)):
        motion_type, body_type, path = paths[i][:3]
        physical_side = paths[i][5]
        if body_type == 'arm':
            path = post_process_motion(physical_chains[physical_side], path, last_qs[physical_side])
            path = [q.tolist() for q in path]
            paths[i][2] = path
            last_qs[physical_side] = path[-1]

    # convert commands and motion from full to active
    for i in range(len(commands)):
        motion_type, body_type, value = commands[i][:3]
        physical_side = commands[i][5]
        if body_type == 'arm':
            if type(value) == tuple:
                q = value[0]
            else:
                q = value
            q_active = physical_chains[physical_side].active_from_full(q).tolist()
            if type(value) == tuple:
                commands[i][2] = (q_active, *value[1:])
            else:
                commands[i][2] = q_active
    for i in range(len(paths)):
        motion_type, body_type, path = paths[i][:3]
        physical_side = paths[i][5]
        if body_type == 'arm':
            paths[i][2] = [physical_chains[physical_side].active_from_full(q).tolist() for q in path]
    clean_commands = [c[:5] for c in commands]
    with open(os.path.join(log_dir, 'commands.pkl'), 'wb') as fp:
        pickle.dump(clean_commands, fp)
    clean_paths = [p[:5] for p in paths]
    with open(os.path.join(log_dir, 'motion.pkl'), 'wb') as fp:
        pickle.dump(clean_paths, fp)

    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'r') as fp:
        stats = json.load(fp)
    stats['motion_plan'] = {'time': round(time() - stamp.start_time, 2)}
    with open(stats_path, 'w') as fp:
        json.dump(stats, fp)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    run_motion_plan(args.assembly_dir, args.log_dir, args.optimized, args.seed, args.verbose)
