import os
import pickle
import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_base_dir)
from planning.robot.util_arm import get_arm_chain

def evaluate_joint_distance(log_dir):
    motion_path = os.path.join(log_dir, 'motion.pkl')
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    if not (os.path.exists(motion_path) and os.path.exists(grasps_path)): return None, None

    with open(grasps_path, 'rb') as f: grasps_data = pickle.load(f)
    with open(motion_path, 'rb') as f: paths = pickle.load(f)

    arm_chains = {
        'right': get_arm_chain(grasps_data.get('arm', 'panda'), 'move', side='right'),
        'left': get_arm_chain(grasps_data.get('arm', 'panda'), 'hold', side='left')
    }

    joint_distances = {'right': 0.0, 'left': 0.0}
    joint_trajectories = {'right': [], 'left': []}

    for segment in paths:
        motion_type, body_type, path_qs = segment[:3]
        physical_side = segment[5] if len(segment) > 5 else ('left' if motion_type == 'hold' else 'right')
        
        if body_type == 'arm' and len(path_qs) > 0:
            chain_side = physical_side if physical_side in arm_chains else 'right'
            prev_q = None
            for q_active in path_qs:
                q_active = np.array(q_active, dtype=float)
                joint_trajectories[chain_side].append(q_active)
                if prev_q is not None:
                    joint_distances[chain_side] += np.linalg.norm(q_active - prev_q)
                prev_q = q_active

    return joint_distances, joint_trajectories

def main(test_name):
    base_dir = os.path.join('logs', test_name)
    if not os.path.exists(base_dir): return
        
    assemblies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    csv_dir = os.path.join('results', test_name, 'csv_reports')
    traj_base_dir = os.path.join('results', test_name, 'joint_trajectories_full')
    os.makedirs(csv_dir, exist_ok=True)
    
    summary_data = []
    
    for assembly in assemblies:
        orig_dist, orig_traj = evaluate_joint_distance(os.path.join(base_dir, assembly, 'original'))
        new_dist, new_traj = evaluate_joint_distance(os.path.join(base_dir, assembly, 'new'))
        
        if orig_dist and new_dist:
            summary_data.append({
                'Assembly': assembly,
                'Joint Dist (Right, rad) (Orig/New)': f"{round(orig_dist['right'], 2)} / {round(new_dist['right'], 2)}",
                'Joint Dist (Left, rad) (Orig/New)': f"{round(orig_dist['left'], 2)} / {round(new_dist['left'], 2)}"
            })

            plot_dir = os.path.join(traj_base_dir, assembly)
            os.makedirs(plot_dir, exist_ok=True)
            
            for side in ['right', 'left']:
                if new_traj[side]:
                    plt.figure(figsize=(10, 5))
                    traj_array = np.array(new_traj[side])
                    for joint_idx in range(traj_array.shape[1]):
                        plt.plot(traj_array[:, joint_idx], label=f'Joint {joint_idx+1}')
                    plt.title(f'{assembly}: {side.capitalize()} Arm Sequence Joint Trajectories')
                    plt.xlabel('Timestep')
                    plt.ylabel('Joint Angle (rad)')
                    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f'{side}_arm_joints.png'), dpi=300)
                    plt.close()

    if summary_data:
        pd.DataFrame(summary_data).to_csv(os.path.join(csv_dir, 'joint_distance_scores.csv'), index=False)
        print(f"Full joint trajectories routed to results/{test_name}/joint_trajectories_full/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type=str)
    args = parser.parse_args()
    main(args.test_name)