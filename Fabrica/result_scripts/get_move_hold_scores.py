import os
import pickle
import sys
import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R
import argparse
import pandas as pd

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_base_dir)

def get_raw_move_score(G_preced, part_move, grasp_move):
    part_contact_points = []
    for predecessor in G_preced.predecessors(part_move):
        part_contact_points.extend(G_preced.edges[predecessor, part_move]['contact_points'])
    part_contact_points = np.array(part_contact_points)
    
    if len(part_contact_points) == 0: return 0.0
        
    grasp_contact_points = np.array(grasp_move.contact_points)
    part_com = G_preced.nodes[part_move]['com']
    path = G_preced.nodes[part_move]['path']
    
    contact_direction = path[-1][:3] - path[0][:3]
    contact_direction /= np.linalg.norm(contact_direction)
    
    part_torque = np.sum(np.cross(part_contact_points - part_com, contact_direction / len(part_contact_points)), axis=0)
    grasp_torque = np.sum(np.cross(grasp_contact_points - part_com, -contact_direction / len(grasp_contact_points)), axis=0)
    net_torque = part_torque + grasp_torque
    
    return np.linalg.norm(net_torque) / len(grasp_contact_points)

def get_raw_hold_score(grasp_hold, grasp_move):
    grasp_score = (R.from_quat(grasp_hold.quat[[1, 2, 3, 0]]).inv() * R.from_quat(grasp_move.quat[[1, 2, 3, 0]])).magnitude()
    return grasp_score * len(grasp_hold.contact_points)

def evaluate_optimal_sequence(log_dir):
    tree_path = os.path.join(log_dir, 'tree_opt.pkl')
    preced_path = os.path.join(log_dir, 'precedence.pkl')
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    
    if not (os.path.exists(tree_path) and os.path.exists(preced_path) and os.path.exists(grasps_path)):
        return None
        
    with open(tree_path, 'rb') as f: tree = pickle.load(f)
    with open(preced_path, 'rb') as f: G_preced = pickle.load(f)
    with open(grasps_path, 'rb') as f: grasps_data = pickle.load(f)
        
    grasps = grasps_data['grasps']
    for part in grasps:
        grasps[part]['move'] = {grasp[0].grasp_id: grasp for grasp in grasps[part]['move']}
        grasps[part]['hold'] = {grasp.grasp_id: grasp for grasp in grasps[part]['hold']}
    
    root_node = [n for n in tree.nodes if tree.in_degree(n) == 0][0]
    cum_move_score = 0.0
    cum_hold_score = 0.0
    
    parent = root_node
    while tree.out_degree(parent) > 0:
        child = list(tree.successors(parent))[0]
        edge_data = tree.edges[parent, child]
        
        move_part = edge_data['move_part']
        hold_part = edge_data['hold_part']
        grasp_move = grasps[move_part]['move'][edge_data['move_grasp_id']][0] 
        grasp_hold = grasps[hold_part]['hold'][edge_data['hold_grasp_id']]
        
        cum_move_score += get_raw_move_score(G_preced, move_part, grasp_move)
        cum_hold_score += get_raw_hold_score(grasp_hold, grasp_move)
        parent = child
        
    return cum_move_score, cum_hold_score

def main(test_name):
    base_dir = os.path.join('logs', test_name)
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return
        
    assemblies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    data_rows = []
    for assembly in assemblies:
        orig_scores = evaluate_optimal_sequence(os.path.join(base_dir, assembly, 'original'))
        new_scores = evaluate_optimal_sequence(os.path.join(base_dir, assembly, 'new'))
        
        if orig_scores and new_scores:
            data_rows.append({
                'Assembly': assembly,
                'Cumulative Move (Raw Torque) (Orig/New)': f"{round(orig_scores[0], 4)} / {round(new_scores[0], 4)}",
                'Cumulative Hold (Raw Rotation) (Orig/New)': f"{round(orig_scores[1], 4)} / {round(new_scores[1], 4)}"
            })

    if not data_rows: return

    # --- ROUTED TO CSV DIRECTORY ---
    results_dir = os.path.join('results', test_name, 'csv_reports')
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(data_rows)
    output_filepath = os.path.join(results_dir, 'raw_scores.csv')
    df.to_csv(output_filepath, index=False)
    print(f"Successfully exported raw scores to {output_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type=str, help='Name of the test directory')
    args = parser.parse_args()
    main(args.test_name)