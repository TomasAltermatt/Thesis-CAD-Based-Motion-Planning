import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_base_dir)
from planning.robot.util_arm import get_arm_chain

def compute_jacobian(arm_chain, q_active, eps=1e-4):
    q_active = np.array(q_active, dtype=float)
    N = len(q_active)
    J = np.zeros((6, N))
    
    def get_full_q(q_act):
        if len(q_act) == len(arm_chain.links): return q_act
        q_full = np.zeros(len(arm_chain.links))
        active_idx = 0
        for i, is_active in enumerate(arm_chain.active_links_mask):
            if is_active and active_idx < len(q_act):
                q_full[i] = q_act[active_idx]
                active_idx += 1
        return q_full

    q_full_0 = get_full_q(q_active)
    T0 = arm_chain.forward_kinematics(q_full_0)
    
    for i in range(N):
        q_plus = q_active.copy()
        q_plus[i] += eps
        q_full_plus = get_full_q(q_plus)
        T_plus = arm_chain.forward_kinematics(q_full_plus)
        p_plus = T_plus[:3, 3]
        R_plus = T_plus[:3, :3]
        J[:3, i] = (p_plus - T0[:3, 3]) / eps
        skew_w_eps = (R_plus @ T0[:3, :3].T) - np.eye(3)
        J[3, i] = skew_w_eps[2, 1] / eps
        J[4, i] = skew_w_eps[0, 2] / eps
        J[5, i] = skew_w_eps[1, 0] / eps
    return J

def evaluate_manipulability(log_dir):
    motion_path = os.path.join(log_dir, 'motion.pkl')
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not (os.path.exists(motion_path) and os.path.exists(grasps_path)): return None

    with open(grasps_path, 'rb') as f: grasps_data = pickle.load(f)
    with open(motion_path, 'rb') as f: paths = pickle.load(f)

    arm_chains = {
        'right': get_arm_chain(grasps_data.get('arm', 'panda'), 'move', side='right'),
        'left': get_arm_chain(grasps_data.get('arm', 'panda'), 'hold', side='left')
    }

    last_q = {'right': None, 'left': None}
    synced_qs = {'right': [], 'left': []}

    for segment in paths:
        motion_type, body_type, path_qs = segment[:3]
        physical_side = segment[5] if len(segment) > 5 else ('left' if motion_type == 'hold' else 'right')
        chain_side = physical_side if physical_side in arm_chains else 'right'
        
        if body_type == 'arm':
            if isinstance(path_qs, np.ndarray) and path_qs.ndim == 1: path_qs = [path_qs]
            elif isinstance(path_qs, list) and len(path_qs) > 0 and isinstance(path_qs[0], (float, int)): path_qs = [path_qs]

            for q in path_qs:
                last_q[chain_side] = q
                if last_q['right'] is not None and last_q['left'] is not None:
                    synced_qs['right'].append(last_q['right'])
                    synced_qs['left'].append(last_q['left'])

    manipulability_scores = {'right': [], 'left': []}
    combined_scores = []

    for i in range(len(synced_qs['right'])):
        Jr = compute_jacobian(arm_chains['right'], synced_qs['right'][i])
        wr = np.sqrt(max(0, np.linalg.det(Jr @ Jr.T)))
        manipulability_scores['right'].append(wr)
        
        Jl = compute_jacobian(arm_chains['left'], synced_qs['left'][i])
        wl = np.sqrt(max(0, np.linalg.det(Jl @ Jl.T)))
        manipulability_scores['left'].append(wl)
        combined_scores.append((wr + wl) / 2.0)

    return {'per_arm': manipulability_scores, 'combined': combined_scores}

def main(test_name):
    base_dir = os.path.join('logs', test_name)
    if not os.path.exists(base_dir): return
        
    assemblies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    csv_dir = os.path.join('results', test_name, 'csv_reports')
    os.makedirs(csv_dir, exist_ok=True)
    summary_data = []
    
    for assembly in assemblies:
        orig_data = evaluate_manipulability(os.path.join(base_dir, assembly, 'original'))
        new_data = evaluate_manipulability(os.path.join(base_dir, assembly, 'new'))
        
        if orig_data and new_data:
            orig_r, orig_l = orig_data['per_arm']['right'], orig_data['per_arm']['left']
            new_r, new_l = new_data['per_arm']['right'], new_data['per_arm']['left']
            orig_comb, new_comb = orig_data['combined'], new_data['combined']

            summary_data.append({
                'Assembly': assembly,
                'Min Manip (Right) (Orig/New)': f"{round(np.min(orig_r), 4)} / {round(np.min(new_r), 4)}",
                'Min Manip (Left) (Orig/New)': f"{round(np.min(orig_l), 4)} / {round(np.min(new_l), 4)}"
            })
            
            plot_dir = os.path.join('results', test_name, 'manipulability_full', assembly)
            os.makedirs(plot_dir, exist_ok=True)

            plt.figure(figsize=(10, 5))
            plt.plot(np.linspace(0, 100, len(orig_comb)), orig_comb, label='Original Pipeline', color='#1f77b4', alpha=0.8)
            plt.plot(np.linspace(0, 100, len(new_comb)), new_comb, label='New Pipeline', color='#ff7f0e', linewidth=2)
            plt.title(f'{assembly}: True System Average Manipulability')
            plt.xlabel('Global Assembly Progress (%)')
            plt.ylabel('Average Yoshikawa Index ($w$)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'overall.png'), dpi=300)
            plt.close()

            plt.figure(figsize=(10, 5))
            if orig_r: plt.plot(np.linspace(0, 100, len(orig_r)), orig_r, label='Orig - Right', color='#1f77b4', linestyle='-')
            if orig_l: plt.plot(np.linspace(0, 100, len(orig_l)), orig_l, label='Orig - Left', color='#1f77b4', linestyle='--')
            if new_r: plt.plot(np.linspace(0, 100, len(new_r)), new_r, label='New - Right', color='#ff7f0e', linestyle='-', linewidth=2)
            if new_l: plt.plot(np.linspace(0, 100, len(new_l)), new_l, label='New - Left', color='#ff7f0e', linestyle='--', linewidth=2)
            plt.title(f'{assembly}: Synchronized Per-Arm Manipulability')
            plt.xlabel('Global Assembly Progress (%)')
            plt.ylabel('Yoshikawa Index ($w$)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'per_arm.png'), dpi=300)
            plt.close()

    if summary_data:
        pd.DataFrame(summary_data).to_csv(os.path.join(csv_dir, 'manipulability_scores.csv'), index=False)
        print(f"Full manipulability plots routed to results/{test_name}/manipulability_full/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type=str)
    args = parser.parse_args()
    main(args.test_name)