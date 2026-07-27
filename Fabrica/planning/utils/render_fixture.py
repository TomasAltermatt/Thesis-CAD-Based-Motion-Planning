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
import trimesh

from utils.renderer import SimRenderer
from assets.load import load_part_ids
from assets.transform import q_to_pos_quat
from planning.robot.geometry import get_gripper_finger_states, get_gripper_finger_states, get_gripper_base_name
from planning.robot.util_arm import get_arm_chain, get_gripper_pos_quat_from_arm_q, get_ik_target_orientation, get_gripper_qm_from_arm_q
from planning.robot.sim_string import arr_to_str, get_color, get_gripper_string, get_arm_string, get_arm_joints
from planning.run_motion_plan import OPEN_RATIO_REST


def create_assembly_dualarm_fixture_xml(assembly_dir, fixture_dir, arm_chain_move, arm_chain_hold, gripper_type, has_ft_sensor, timestep=1e-3):
    arm_quat_move = R.from_euler('xyz', arm_chain_move.base_euler).as_quat()[[3, 0, 1, 2]]
    arm_quat_hold = R.from_euler('xyz', arm_chain_hold.base_euler).as_quat()[[3, 0, 1, 2]]
    rest_q_move = arm_chain_move.active_to_full(arm_chain_move.rest_q)
    rest_q_hold = arm_chain_hold.active_to_full(arm_chain_hold.rest_q)
    gripper_pos_move, gripper_quat_move = get_gripper_pos_quat_from_arm_q(arm_chain_move, rest_q_move, gripper_type, has_ft_sensor=has_ft_sensor['move'])
    gripper_pos_hold, gripper_quat_hold = get_gripper_pos_quat_from_arm_q(arm_chain_hold, rest_q_hold, gripper_type, has_ft_sensor=has_ft_sensor['hold'])
    arm_type = arm_chain_move.arm_type # NOTE: assume move and hold have the same arm type
    board_quat = "0.7073883 0 0 0.7068252" if arm_type == 'panda' else "1 0 0 0"
    with open(os.path.join(fixture_dir, 'pickup.json'), 'r') as f:
        part_pickup_pose = json.load(f)
    string = f'''
<redmax model="dual_gripper_arm">
<option integrator="BDF1" timestep="{timestep}" gravity="0. 0. 1e-12"/>
<ground pos="0 0 -1.0" normal="0 0 1"/>
'''
    for part_id in load_part_ids(assembly_dir):
        pos, quat = q_to_pos_quat(part_pickup_pose[part_id])
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="free3d-exp" axis="0. 0. 0." pos="{arr_to_str(pos)}" quat="{arr_to_str(quat)}" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="mesh" filename="{assembly_dir}/{part_id}.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="{arr_to_str(get_color(part_id))}"/>
    </link>
</robot>
'''
    string += f'''
<robot>
    <link name="optical_board">
        <joint name="optical_board" type="fixed" axis="0. 0. 0." pos="0 0 0" quat="{board_quat}" frame="WORLD" damping="0"/>
        <body name="optical_board" type="mesh" filename="assets/optical_board.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="0.2 0.2 0.2 1.0"/>
    </link>
</robot>
'''
    string += f'''
<robot>
    <link name="fixture">
        <joint name="fixture" type="fixed" axis="0. 0. 0." pos="0 0 0" quat="1 0 0 0" frame="WORLD" damping="0"/>
        <body name="fixture" type="mesh" filename="{fixture_dir}/fixture.obj" pos="0 0 0" quat="1 0 0 0" transform_type="OBJ_TO_JOINT" density="1" mu="0" rgba="0.5 0.5 0.5 1.0"/>
    </link>
</robot>
'''
    string += get_gripper_string(gripper_type, gripper_pos_move, gripper_quat_move, fixed=False, has_ft_sensor=has_ft_sensor['move'], suffix='move')
    string += get_gripper_string(gripper_type, gripper_pos_hold, gripper_quat_hold, fixed=False, has_ft_sensor=has_ft_sensor['hold'], suffix='hold')
    string += get_arm_string(arm_chain_move.arm_type, arm_chain_move.base_pos, arm_quat_move, suffix='move')
    string += get_arm_string(arm_chain_hold.arm_type, arm_chain_hold.base_pos, arm_quat_hold, suffix='hold')
    string += f'''
</redmax>
    '''
    return string


def get_camera_option(option):
    if option == 0: # close
        camera_lookat = [-1, 1, 0]
        camera_pos = [1.25, -1.5, 1.5]
    elif option == 1: # far
        camera_lookat = [4.73023, -3.64037, 5.60194]
        camera_pos = [5.34119, -4.31921, 6.00925]
    else:
        raise NotImplementedError
    return camera_lookat, camera_pos


def render_fixture(assembly_dir, log_dir, move_q=None, hold_q=None, move_pos=None, hold_pos=None, move_open_ratio=None, hold_open_ratio=None):

    asset_folder = os.path.join(project_base_dir, './assets')

    with open(os.path.join(log_dir, 'grasps.pkl'), 'rb') as fp:
        grasps = pickle.load(fp)
    arm_type, gripper_type, has_ft_sensor = grasps['arm'], grasps['gripper'], grasps['ft_sensor']
    
    fixture_dir = os.path.join(log_dir, 'fixture')
    with open(os.path.join(fixture_dir, 'pickup.json'), 'r') as f:
        part_pickup_pose = json.load(f)

    # part_meshes = load_part_meshes(assembly_dir, transform='none')
    # part_meshes = {k[4:]: v for k, v in part_meshes.items()}
    # for part_id, part_mesh in part_meshes.items():
    #     part_mesh.apply_transform(get_transform_matrix(part_pickup_pose[part_id]))
    # fixture_mesh = trimesh.load_mesh(os.path.join(fixture_dir, 'fixture.obj'))
    # trimesh.Scene(list(part_meshes.values()) + [fixture_mesh]).show()

    arm_chain_move = get_arm_chain(arm_type, 'move')
    arm_chain_hold = get_arm_chain(arm_type, 'hold')
    arm_joints = get_arm_joints(arm_type)

    # initialize simulation
    xml_string = create_assembly_dualarm_fixture_xml(assembly_dir, fixture_dir, arm_chain_move, arm_chain_hold, gripper_type, has_ft_sensor)
    sim = redmax.Simulation(xml_string, asset_folder) # order: parts, gripper move, gripper hold, arm move, arm hold

    camera_lookat, camera_pos = get_camera_option(1)
    sim.viewer_options.camera_lookat = camera_lookat
    sim.viewer_options.camera_pos = camera_pos

    if move_q is None:
        if move_pos is not None:
            move_euler = np.array([np.pi, 0, -np.pi / 2])
            move_ori = get_ik_target_orientation(arm_chain_move, gripper_type, R.from_euler('xyz', move_euler).as_quat()[[3, 0, 1, 2]])
            move_q, ik_success = arm_chain_move.inverse_kinematics(target_position=move_pos, target_orientation=move_ori, orientation_mode='all', optimizer='least_squares')
            assert ik_success, 'Failed to find IK solution for move arm'
            move_q = arm_chain_move.active_from_full(move_q)
        else:
            move_q = arm_chain_move.rest_q
    if hold_q is None:
        if hold_pos is not None:
            hold_euler = np.array([np.pi, 0, -np.pi / 2])
            hold_ori = get_ik_target_orientation(arm_chain_hold, gripper_type, R.from_euler('xyz', hold_euler).as_quat()[[3, 0, 1, 2]])
            hold_q, ik_success = arm_chain_hold.inverse_kinematics(target_position=hold_pos, target_orientation=hold_ori, orientation_mode='all', optimizer='least_squares')
            assert ik_success, 'Failed to find IK solution for hold arm'
            hold_q = arm_chain_hold.active_from_full(hold_q)
        else:
            hold_q = arm_chain_hold.rest_q
    if move_open_ratio is None: move_open_ratio = OPEN_RATIO_REST
    if hold_open_ratio is None: hold_open_ratio = OPEN_RATIO_REST
    
    # set initial joint states
    for arm_joint_name, arm_joint_state in zip(arm_joints, move_q):
        sim.set_joint_q_init(arm_joint_name + '_move', np.array([arm_joint_state]))
    for arm_joint_name, arm_joint_state in zip(arm_joints, hold_q):
        sim.set_joint_q_init(arm_joint_name + '_hold', np.array([arm_joint_state]))
    gripper_move_name = get_gripper_base_name(gripper_type, suffix='move')
    move_q_full = arm_chain_move.active_to_full(move_q)
    gripper_move_qm = get_gripper_qm_from_arm_q(arm_chain_move, move_q_full, gripper_type, has_ft_sensor=has_ft_sensor['move'])
    gripper_move_q = sim.get_joint_q_from_qm(gripper_move_name, gripper_move_qm)
    sim.set_joint_q_init(gripper_move_name, gripper_move_q)
    gripper_hold_name = get_gripper_base_name(gripper_type, suffix='hold')
    hold_q_full = arm_chain_hold.active_to_full(hold_q)
    gripper_hold_qm = get_gripper_qm_from_arm_q(arm_chain_hold, hold_q_full, gripper_type, has_ft_sensor=has_ft_sensor['hold'])
    gripper_hold_q = sim.get_joint_q_from_qm(gripper_hold_name, gripper_hold_qm)
    sim.set_joint_q_init(gripper_hold_name, gripper_hold_q)
    finger_move_states = get_gripper_finger_states(gripper_type, move_open_ratio, suffix='move')
    for finger_name, finger_state in finger_move_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    finger_hold_states = get_gripper_finger_states(gripper_type, hold_open_ratio, suffix='hold')
    for finger_name, finger_state in finger_hold_states.items():
        sim.set_joint_q_init(finger_name, np.array(finger_state))
    sim.reset(backward_flag=False)

    SimRenderer.replay(sim)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--move-q', type=float, nargs='+', default=None)
    parser.add_argument('--hold-q', type=float, nargs='+', default=None)
    parser.add_argument('--move-pos', type=float, nargs=3, default=None)
    parser.add_argument('--hold-pos', type=float, nargs=3, default=None)
    parser.add_argument('--move-open-ratio', type=float, default=None)
    parser.add_argument('--hold-open-ratio', type=float, default=None)
    args = parser.parse_args()
    
    render_fixture(args.assembly_dir, args.log_dir,
        move_q=args.move_q, hold_q=args.hold_q, 
        move_pos=args.move_pos, hold_pos=args.hold_pos, 
        move_open_ratio=args.move_open_ratio, hold_open_ratio=args.hold_open_ratio)
