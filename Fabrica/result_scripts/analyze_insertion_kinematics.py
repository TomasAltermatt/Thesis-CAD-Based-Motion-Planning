import os
import sys
import pickle
import numpy as np
import argparse

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)
from planning.robot.util_arm import get_arm_chain

def compute_jacobian(arm_chain, q_active, eps=1e-4):
    """Computes a 6xN numerical Jacobian using forward kinematics."""
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

def analyze_insertion_kinematics(log_dir):
    motion_path = os.path.join(log_dir, 'motion.pkl')
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    
    if not (os.path.exists(motion_path) and os.path.exists(grasps_path)): return False

    with open(motion_path, 'rb') as f: motions = pickle.load(f)
    with open(grasps_path, 'rb') as f: grasps_data = pickle.load(f)

    arm_type = grasps_data.get('arm', 'panda')
    arm_chains = {
        'right': get_arm_chain(arm_type, 'move', side='right'),
        'left': get_arm_chain(arm_type, 'hold', side='left')
    }

    current_q = {'right': None, 'left': None}
    insertion_data = {}

    for motion in motions:
        motion_type, body_type, path, active_part, task = motion[:5]
        physical_side = motion[5] if len(motion) == 6 else ('left' if motion_type == 'hold' else 'right')

        if body_type == 'arm': 
            if isinstance(path, np.ndarray) and path.ndim == 1: path = [path]
            elif isinstance(path, list) and len(path) > 0 and isinstance(path[0], (float, int)): path = [path]
            if len(path) > 0: current_q[physical_side] = path[-1]

        # --- NEW: Catch the Base Part Placement ---
        if task == 'transport' and active_part is not None and len(insertion_data) == 0:
            move_side = physical_side
            hold_side = 'left' if move_side == 'right' else 'right'
            hold_q = current_q[hold_side] 
            
            w_move_list, w_hold_list = [], []
            approach_path = path[-15:] if len(path) > 15 else path # Grab last 15 frames of transport
            
            for q_move in approach_path:
                J_move = compute_jacobian(arm_chains[move_side], q_move)
                w_move_list.append(np.sqrt(max(0, np.linalg.det(J_move @ J_move.T))))
                if hold_q is not None:
                    J_hold = compute_jacobian(arm_chains[hold_side], hold_q)
                    w_hold_list.append(np.sqrt(max(0, np.linalg.det(J_hold @ J_hold.T))))
                else:
                    w_hold_list.append(0.0)

            insertion_data[active_part] = {
                'move_side': move_side, 'hold_side': hold_side, 'timesteps': len(approach_path),
                'move_joint_angles': approach_path, 'hold_joint_angle': hold_q,
                'move_manipulability': w_move_list, 'hold_manipulability': w_hold_list, 
                'is_base_part': True
            }

        # --- Original: Catch standard straight-line insertions ---
        elif task == 'assembly' and body_type == 'arm':
            move_side = physical_side
            hold_side = 'left' if move_side == 'right' else 'right'
            hold_q = current_q[hold_side] 
            
            w_move_list, w_hold_list = [], []
            for q_move in path:
                J_move = compute_jacobian(arm_chains[move_side], q_move)
                w_move_list.append(np.sqrt(max(0, np.linalg.det(J_move @ J_move.T))))
                if hold_q is not None:
                    J_hold = compute_jacobian(arm_chains[hold_side], hold_q)
                    w_hold_list.append(np.sqrt(max(0, np.linalg.det(J_hold @ J_hold.T))))
                else:
                    w_hold_list.append(0.0)

            insertion_data[active_part] = {
                'move_side': move_side, 'hold_side': hold_side, 'timesteps': len(path),
                'move_joint_angles': path, 'hold_joint_angle': hold_q,
                'move_manipulability': w_move_list, 'hold_manipulability': w_hold_list, 
                'is_base_part': False
            }

    with open(os.path.join(log_dir, 'insertion_analysis.pkl'), 'wb') as f:
        pickle.dump(insertion_data, f)
    return True

def main(test_name):
    base_dir = os.path.join('logs', test_name)
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return
        
    assemblies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for assembly in assemblies:
        for variant in ['original', 'new']:
            variant_dir = os.path.join(base_dir, assembly, variant)
            if analyze_insertion_kinematics(variant_dir):
                print(f"Processed insertion segments for: {assembly} ({variant})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type=str, help='Name of the test directory')
    args = parser.parse_args()
    main(args.test_name)