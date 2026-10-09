import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

def get_joint_distances(log_dir):
    """Calculates total joint travel for Right and Left arms from motion.pkl"""
    motion_path = os.path.join(log_dir, 'motion.pkl')
    if not os.path.exists(motion_path): return None, None
    
    dist_r, dist_l = 0.0, 0.0
    with open(motion_path, 'rb') as f: motions = pickle.load(f)
        
    for m in motions:
        if len(m) < 3: continue
        body = m[1]
        path = m[2]
        side = m[5] if len(m) == 6 else ('left' if m[0] == 'hold' else 'right')
        
        if body == 'arm' and len(path) > 1:
            if isinstance(path, np.ndarray) and path.ndim == 1: continue
            
            travel = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) for i in range(1, len(path)))
            if side == 'right':
                dist_r += travel
            else:
                dist_l += travel
                
    return dist_r, dist_l

def get_move_hold_scores(log_dir):
    """Calculates cumulative Move Torque and Hold Scores from tree_opt.pkl and grasps.pkl"""
    tree_path = os.path.join(log_dir, 'tree_opt.pkl')
    grasps_path = os.path.join(log_dir, 'grasps.pkl')
    preced_path = os.path.join(log_dir, 'precedence.pkl')
    
    if not (os.path.exists(tree_path) and os.path.exists(grasps_path) and os.path.exists(preced_path)):
        return None, None
        
    with open(tree_path, 'rb') as f: tree = pickle.load(f)
    with open(grasps_path, 'rb') as f: grasps = pickle.load(f)
    with open(preced_path, 'rb') as f: G_preced = pickle.load(f)

    sum_move, sum_hold = 0.0, 0.0
    
    # Format grasp dict for fast lookup
    if 'grasps' in grasps and isinstance(grasps['grasps'], dict):
        g_dict = grasps['grasps']
        for part in g_dict:
            if isinstance(g_dict[part]['move'], list):
                g_dict[part]['move'] = {g[0].grasp_id: g for g in g_dict[part]['move']}
            if isinstance(g_dict[part]['hold'], list):
                g_dict[part]['hold'] = {g.grasp_id: g for g in g_dict[part]['hold']}
    else:
        g_dict = grasps

    try:
        root_node = [n for n in tree.nodes if tree.in_degree(n) == 0][0]
        curr_node = root_node
        
        while tree.out_degree(curr_node) > 0:
            child_node = list(tree.successors(curr_node))[0]
            edge = tree.edges[curr_node, child_node]
            
            m_part, h_part = edge['move_part'], edge['hold_part']
            m_grasp_id, h_grasp_id = edge['move_grasp_id'], edge['hold_grasp_id']
            
            m_grasp = g_dict[m_part]['move'][m_grasp_id][0]
            h_grasp = g_dict[h_part]['hold'][h_grasp_id]
            
            # 1. Move Score (Geometric Torque Proxy)
            part_pts = []
            for pred in G_preced.predecessors(m_part):
                part_pts.extend(G_preced.edges[pred, m_part]['contact_points'])
            
            if len(part_pts) > 0:
                part_pts = np.array(part_pts)
                grasp_pts = np.array(m_grasp.contact_points)
                part_com = G_preced.nodes[m_part]['com']
                path = G_preced.nodes[m_part]['path']
                
                contact_dir = path[-1][:3] - path[0][:3]
                contact_dir /= np.linalg.norm(contact_dir)
                
                p_torque = np.sum(np.cross(part_pts - part_com, contact_dir / len(part_pts)), axis=0)
                g_torque = np.sum(np.cross(grasp_pts - part_com, -contact_dir / len(grasp_pts)), axis=0)
                sum_move += np.linalg.norm(p_torque + g_torque) / len(grasp_pts)

            # 2. Hold Score (Angle * Contacts)
            angle_diff = (R.from_quat(h_grasp.quat[[1, 2, 3, 0]]).inv() * R.from_quat(m_grasp.quat[[1, 2, 3, 0]])).magnitude()
            sum_hold += angle_diff * len(h_grasp.contact_points)
            
            curr_node = child_node
    except Exception as e:
        print(f"Warning: Score extraction failed for {log_dir}. Error: {e}")
        return None, None

    return sum_move, sum_hold

def extract_manipulability(log_dir):
    """Extracts the averaged manipulability curve from insertion_analysis.pkl"""
    analysis_path = os.path.join(log_dir, 'insertion_analysis.pkl')
    if not os.path.exists(analysis_path): return None
    
    with open(analysis_path, 'rb') as f: data = pickle.load(f)
    
    x_common = np.linspace(0, 100, 101)
    interp_curves = []
    
    for part, p_data in data.items():
        w_move = p_data['move_manipulability']
        w_hold = p_data['hold_manipulability']
        if len(w_move) == 0: continue
            
        x_orig = np.linspace(0, 100, len(w_move))
        total_w = np.array(w_move) + np.array(w_hold)
        interp_curves.append(np.interp(x_common, x_orig, total_w))
        
    if len(interp_curves) == 0: return None
    return np.mean(interp_curves, axis=0)

def main():
    print("\n" + "="*55)
    print("   DYNAMIC ABLATION STUDY GENERATOR (FORMATTED DELTAS)")
    print("="*55)
    
    # 1. Initialization
    study_name = input("Enter a name for this Ablation Study (e.g., 'kinematics_weights'): ").strip()
    if not study_name: study_name = "ablation_study"
        
    output_dir = os.path.join('results', 'ablation_studies', study_name)
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = []
    print("\n[INPUT INSTRUCTIONS]")
    print("1. The FIRST experiment you enter MUST contain the 'original' baseline logs.")
    print("2. For subsequent experiments, only the 'new' folders will be evaluated.")
    print("3. Leave the Folder Name blank and press Enter to finish inputting.\n")
    
    while True:
        folder = input(f"Experiment {len(experiments)+1} Folder Name (inside 'logs/'): ").strip()
        if not folder:
            if len(experiments) == 0:
                print("Error: You must enter at least one experiment.")
                continue
            break
            
        label = input(f"Enter Legend Label for '{folder}' (e.g., '100% Weights'): ").strip()
        if not label: label = folder
        experiments.append({'folder': folder, 'label': label})

    base_exp = experiments[0]['folder']
    base_dir = os.path.join('logs', base_exp)
    if not os.path.exists(base_dir):
        print(f"\nError: Base experiment '{base_dir}' not found. Exiting.")
        return

    assemblies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    joint_rows = []
    score_rows = []
    plot_data = {asm: {} for asm in assemblies}

    print(f"\nAnalyzing logs, calculating absolute deltas, and formatting {len(assemblies)} assemblies...")
    
    for asm in assemblies:
        # 1. Fetch ORIGINAL Baseline
        orig_dir = os.path.join('logs', base_exp, asm, 'original')
        orig_r, orig_l = get_joint_distances(orig_dir)
        orig_m, orig_h = get_move_hold_scores(orig_dir)
        plot_data[asm]['Original'] = extract_manipulability(orig_dir)

        # Initialize formatted rows
        j_row = {'Assembly': asm.upper()}
        s_row = {'Assembly': asm.upper()}
        
        if orig_r is not None and orig_l is not None:
            j_row['Orig (Right / Left)'] = f"{orig_r:.2f} / {orig_l:.2f}"
        else:
            j_row['Orig (Right / Left)'] = "N/A"
            
        if orig_m is not None and orig_h is not None:
            s_row['Orig (Move / Hold)'] = f"{orig_m:.3f} / {orig_h:.2f}"
        else:
            s_row['Orig (Move / Hold)'] = "N/A"

        # 2. Iterate through NEW Studies
        for exp in experiments:
            new_dir = os.path.join('logs', exp['folder'], asm, 'new')
            new_r, new_l = get_joint_distances(new_dir)
            new_m, new_h = get_move_hold_scores(new_dir)
            
            lbl = exp['label']
            plot_data[asm][lbl] = extract_manipulability(new_dir)
            
            # Format Joint Differences with explicit signs (+/-)
            if new_r is not None and orig_r is not None:
                j_row[f"{lbl} Diff (Right / Left)"] = f"{(new_r - orig_r):+.2f} / {(new_l - orig_l):+.2f}"
            else:
                j_row[f"{lbl} Diff (Right / Left)"] = "N/A"
                
            # Format Score Differences with explicit signs (+/-)
            if new_m is not None and orig_m is not None:
                s_row[f"{lbl} Diff (Move / Hold)"] = f"{(new_m - orig_m):+.3f} / {(new_h - orig_h):+.2f}"
            else:
                s_row[f"{lbl} Diff (Move / Hold)"] = "N/A"
            
        joint_rows.append(j_row)
        score_rows.append(s_row)

    # Save cleanly formatted CSVs
    df_joints = pd.DataFrame(joint_rows)
    df_scores = pd.DataFrame(score_rows)

    csv_joints_path = os.path.join(output_dir, 'ablation_joint_differences.csv')
    csv_scores_path = os.path.join(output_dir, 'ablation_score_differences.csv')
    df_joints.to_csv(csv_joints_path, index=False)
    df_scores.to_csv(csv_scores_path, index=False)

    # --- PLOTTING LOGIC ---
    valid_assemblies = [asm for asm in assemblies if plot_data[asm].get('Original') is not None]
    if not valid_assemblies:
        print("\nWarning: No insertion_analysis.pkl files found. Did you run the kinematics analyzer first?")
        return

    n = len(valid_assemblies)
    cols = min(3, n)
    rows = math.ceil(n / cols) if n > 0 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    x_common = np.linspace(0, 100, 101)
    
    # Distinct colors for ablation studies
    color_map = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i, asm in enumerate(valid_assemblies):
        ax = axes[i]
        
        # 1. Plot Baseline (Dashed Blue)
        ax.plot(x_common, plot_data[asm]['Original'], color='#1f77b4', linestyle='--', linewidth=2.5, alpha=0.8, label='Original')
        
        # 2. Plot New Studies (Solid Colors)
        for j, exp in enumerate(experiments):
            lbl = exp['label']
            if plot_data[asm].get(lbl) is not None:
                color = color_map[j % len(color_map)]
                ax.plot(x_common, plot_data[asm][lbl], color=color, linestyle='-', linewidth=3.0, label=lbl)
            
        ax.set_title(asm.upper(), fontsize=14, fontweight='bold')
        ax.set_xlabel("Insertion Progress (%)", fontsize=12)
        ax.set_ylabel("Avg Total Manipulability ($w_{move} + w_{hold}$)", fontsize=12)
        ax.grid(True, alpha=0.4)
        ax.legend()
        
    for i in range(len(valid_assemblies), len(axes)):
        axes[i].set_visible(False)
        
    fig.suptitle(f"Ablation Study Kinematics: {study_name.replace('_', ' ').title()}", fontsize=20, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.92 if rows > 1 else 0.85)
    
    plot_path = os.path.join(output_dir, 'global_ablation_kinematics.png')
    plt.savefig(plot_path, dpi=300)
    
    print("\n" + "="*55)
    print(f"SUCCESS! Formatted files saved to: {output_dir}/")
    print("  - ablation_joint_differences.csv")
    print("  - ablation_score_differences.csv")
    print("  - global_ablation_kinematics.png")
    print("="*55)

if __name__ == '__main__':
    main()