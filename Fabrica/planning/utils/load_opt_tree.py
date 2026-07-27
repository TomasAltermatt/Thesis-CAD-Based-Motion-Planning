import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import pickle
import networkx as nx

import numpy as np
from assets.load import load_assembly_all_transformed, load_pos_quat_dict
from planning.robot.geometry import get_gripper_finger_names, transform_gripper_meshes, load_part_meshes, load_gripper_meshes, get_part_meshes_transforms
from planning.robot.workcell import get_assembly_center


def compute_contact_area(part_meshes, part_final_transforms, gripper_type, gripper_meshes, grasp, part_id, n_sample=5000):
    part_mesh = part_meshes[part_id].copy()
    part_mesh.apply_transform(part_final_transforms[part_id])

    gripper_pos, gripper_quat, open_ratio = grasp.pos, grasp.quat, grasp.open_ratio
    gripper_finger_meshes = {name: mesh.copy() for name, mesh in gripper_meshes.items() if name in get_gripper_finger_names(gripper_type)}
    gripper_finger_meshes = transform_gripper_meshes(gripper_type, gripper_finger_meshes, gripper_pos, gripper_quat, np.eye(4), open_ratio - 0.01) # NOTE: 0.01 for slight penetration

    contact_area = 0
    for name, mesh in gripper_finger_meshes.items():
        contact_area += part_mesh.contains(mesh.sample(n_sample)).sum() / n_sample * mesh.area

    return contact_area


def get_move_grasp_score(G_preced, grasp_move):
    move_part = grasp_move.part_id
    part_contact_points = []
    for predecessor in G_preced.predecessors(move_part):
        part_contact_points.extend(G_preced.edges[predecessor, move_part]['contact_points'])
    part_contact_points = np.array(part_contact_points)
    if len(part_contact_points) == 0:
        return 0
    grasp_contact_points = np.array(grasp_move.contact_points)
    part_com = G_preced.nodes[move_part]['com']
    path = G_preced.nodes[move_part]['path']
    contact_direction = path[-1][:3] - path[0][:3]
    contact_direction /= np.linalg.norm(contact_direction)
    part_torque = np.sum(np.cross(part_contact_points - part_com, contact_direction / len(part_contact_points)), axis=0)
    grasp_torque = np.sum(np.cross(grasp_contact_points - part_com, -contact_direction / len(grasp_contact_points)), axis=0)
    net_torque = part_torque + grasp_torque
    grasp_score = np.linalg.norm(net_torque) / len(grasp_contact_points)
    return grasp_score


def find_root_node(tree):
    for node in tree.nodes:
        if tree.in_degree(node) == 0:
            return node


def load_opt_tree(assembly_dir, log_dir):
    precedence_path = os.path.join(log_dir, 'precedence.pkl')
    with open(precedence_path, 'rb') as f:
        G_preced = pickle.load(f)
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    with open(grasps_path, 'rb') as f:
        grasps_data = pickle.load(f)
    gripper_type, arm_type = grasps_data['gripper'], grasps_data['arm']

    assembly = load_assembly_all_transformed(assembly_dir)
    part_ids = list(assembly.keys())
    part_meshes = {part_id: assembly[part_id]['mesh'] for part_id in part_ids}
    part_pos_dict, part_quat_dict = load_pos_quat_dict(assembly_dir, transform='final')
    part_pos_dict = {part_id: part_pos_dict[part_id] + get_assembly_center(arm_type) for part_id in part_ids}
    part_final_transforms = get_part_meshes_transforms(part_meshes, part_pos_dict, part_quat_dict, np.eye(4))
    asset_folder = os.path.join(project_base_dir, 'assets')
    gripper_meshes = load_gripper_meshes(gripper_type, asset_folder)

    grasps = grasps_data['grasps']
    parts = list(grasps.keys())
    for part in parts: # convert list to dict for faster search
        grasps[part]['move'] = {grasp[0].grasp_id: grasp for grasp in grasps[part]['move']}
        grasps[part]['hold'] = {grasp.grasp_id: grasp for grasp in grasps[part]['hold']}

    opt_tree_path = os.path.join(log_dir, 'tree_opt.pkl')
    with open(opt_tree_path, 'rb') as f:
        opt_tree = pickle.load(f)

    root_node = find_root_node(opt_tree)
    
    parent_node = root_node
    while opt_tree.out_degree(parent_node) > 0:
        children_nodes = list(opt_tree.successors(parent_node))
        assert len(children_nodes) == 1
        child_node = children_nodes[0]
        edge_info = opt_tree.edges[parent_node, child_node]
        move_part, hold_part = edge_info['move_part'], edge_info['hold_part']
        move_grasp_id, hold_grasp_id = edge_info['move_grasp_id'], edge_info['hold_grasp_id']
        print(f'move part: {move_part}, hold part: {hold_part}, move grasp id: {move_grasp_id}, hold grasp id: {hold_grasp_id}')
        move_grasp, hold_grasp = grasps[move_part]['move'][move_grasp_id], grasps[hold_part]['hold'][hold_grasp_id]
        # print(f'move grasp area: {move_grasp[0].contact_area}, hold grasp area: {hold_grasp.contact_area}')
        move_area = compute_contact_area(part_meshes, part_final_transforms, gripper_type, gripper_meshes, move_grasp[0], move_part)
        hold_area = compute_contact_area(part_meshes, part_final_transforms, gripper_type, gripper_meshes, hold_grasp, hold_part)
        print(f'move grasp area: {move_area}, hold grasp area: {hold_area}')
        # print(f'move part area: {assembly[move_part]["mesh"].area}, hold part area: {assembly[hold_part]["mesh"].area}')
        print(f'move grasp score: {get_move_grasp_score(G_preced, move_grasp[0])}')
        parent_node = child_node


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    args = parser.parse_args()
    
    load_opt_tree(args.assembly_dir, args.log_dir)
