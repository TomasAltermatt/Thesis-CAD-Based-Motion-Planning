import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

def generate_comparative_plots(base_dir, test_name, assembly_name, global_orig_dict, global_new_dict):
    orig_path = os.path.join(base_dir, assembly_name, 'original', 'insertion_analysis.pkl')
    new_path = os.path.join(base_dir, assembly_name, 'new', 'insertion_analysis.pkl')
    
    orig_data, new_data = {}, {}
    if os.path.exists(orig_path):
        with open(orig_path, 'rb') as f: orig_data = pickle.load(f)
    if os.path.exists(new_path):
        with open(new_path, 'rb') as f: new_data = pickle.load(f)
        
    parts = list(dict.fromkeys(list(orig_data.keys()) + list(new_data.keys())))
    if not parts: return

    base_results_dir = os.path.join('results', test_name, 'insertion_plots')
    overlay_dir = os.path.join(base_results_dir, 'overlay_comparisons')
    subplots_dir = os.path.join(base_results_dir, 'subplot_comparisons')
    individual_dir = os.path.join(base_results_dir, 'individual_parts', assembly_name)
    
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(subplots_dir, exist_ok=True)
    os.makedirs(individual_dir, exist_ok=True)

    def plot_lines_for_part(ax, data, part, is_new):
        if part not in data: return
        w_move = data[part]['move_manipulability']
        w_hold = data[part]['hold_manipulability']
        x_norm = np.linspace(0, 100, len(w_move))
        
        color = '#ff7f0e' if is_new else '#1f77b4'
        prefix = 'New' if is_new else 'Orig'
        
        ax.plot(x_norm, w_move, color=color, linewidth=2.5, label=f"{prefix} Move ({data[part]['move_side'].upper()})")
        ax.plot(x_norm, w_hold, color=color, linestyle='--', linewidth=2.5, alpha=0.7, label=f"{prefix} Hold ({data[part]['hold_side'].upper()})")

    # Common X-axis for interpolating the assembly averages
    x_common = np.linspace(0, 100, 101)
    assembly_orig_interp = []
    assembly_new_interp = []

    # 1. OVERLAY MODE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(parts)))
    
    for i, part in enumerate(parts):
        if part in orig_data:
            w_m, w_h = orig_data[part]['move_manipulability'], orig_data[part]['hold_manipulability']
            x = np.linspace(0, 100, len(w_m))
            line, = ax1.plot(x, w_m, color=colors[i], linewidth=2, label=part, alpha=0.8)
            ax1.plot(x, w_h, '--', color=line.get_color(), linewidth=2, alpha=0.4)
            
            # Accumulate Total Manipulability for this assembly's average
            total_w = np.array(w_m) + np.array(w_h)
            assembly_orig_interp.append(np.interp(x_common, x, total_w))
            
        if part in new_data:
            w_m, w_h = new_data[part]['move_manipulability'], new_data[part]['hold_manipulability']
            x = np.linspace(0, 100, len(w_m))
            line, = ax2.plot(x, w_m, color=colors[i], linewidth=2, label=part, alpha=0.8)
            ax2.plot(x, w_h, '--', color=line.get_color(), linewidth=2, alpha=0.4)
            
            # Accumulate Total Manipulability for this assembly's average
            total_w = np.array(w_m) + np.array(w_h)
            assembly_new_interp.append(np.interp(x_common, x, total_w))

    # Store the final averaged arrays for the global plot
    if assembly_orig_interp:
        global_orig_dict[assembly_name] = np.mean(assembly_orig_interp, axis=0)
    if assembly_new_interp:
        global_new_dict[assembly_name] = np.mean(assembly_new_interp, axis=0)

    fig.suptitle(f"Insertion Stroke Manipulability: {assembly_name.upper()}", fontsize=18, fontweight='bold')
    ax1.set_title("ORIGINAL PIPELINE", fontsize=14, fontweight='bold')
    ax2.set_title("NEW PIPELINE", fontsize=14, fontweight='bold')
    for ax in [ax1, ax2]:
        ax.set_xlabel("Insertion Progress (%)", fontsize=12)
        ax.set_ylabel("Yoshikawa Index (w)", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(overlay_dir, f'{assembly_name}_overlay_comparison.png'), dpi=300)
    plt.close()

    # 2. SUBPLOTS MODE
    n = len(parts)
    cols = min(3, n)
    rows = math.ceil(n / cols) if n > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i, part in enumerate(parts):
        is_base = (part in orig_data and orig_data[part].get('is_base_part')) or \
                  (part in new_data and new_data[part].get('is_base_part'))
        
        plot_lines_for_part(axes[i], orig_data, part, is_new=False)
        plot_lines_for_part(axes[i], new_data, part, is_new=True)
        
        title = f"Base Part Placement: {part}" if is_base else f"Insertion: {part}"
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Stroke Progress (%)")
        axes[i].set_ylabel("Yoshikawa Index (w)")
        axes[i].grid(True, alpha=0.4)
        axes[i].legend()
        
    for i in range(len(parts), len(axes)): axes[i].set_visible(False)
    
    fig.suptitle(f"Detailed Part Kinematics: {assembly_name.upper()}", fontsize=20, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.92 if rows > 1 else 0.85)
    plt.savefig(os.path.join(subplots_dir, f'{assembly_name}_subplots_grid.png'), dpi=300)
    plt.close()

    # 3. INDIVIDUAL MODE
    for part in parts:
        is_base = (part in orig_data and orig_data[part].get('is_base_part')) or \
                  (part in new_data and new_data[part].get('is_base_part'))
                  
        plt.figure(figsize=(9, 6))
        plot_lines_for_part(plt.gca(), orig_data, part, is_new=False)
        plot_lines_for_part(plt.gca(), new_data, part, is_new=True)
        
        title_prefix = "Base Part Placement" if is_base else "Insertion Kinematics"
        plt.title(f"{title_prefix}: {part} ({assembly_name.upper()})", fontsize=14, fontweight='bold')
        plt.xlabel("Stroke Progress (%)", fontsize=12)
        plt.ylabel("Yoshikawa Index (w)", fontsize=12)
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        
        filename = f"{part.replace('/', '_').replace(' ', '_')}_comparison.png"
        plt.savefig(os.path.join(individual_dir, filename), dpi=300)
        plt.close()

def generate_global_summary_plot(test_name, global_orig_dict, global_new_dict):
    """Generates a grid of subplots showing the average system manipulability per assembly."""
    all_assemblies = sorted(list(set(list(global_orig_dict.keys()) + list(global_new_dict.keys()))))
    if not all_assemblies: return
    
    n = len(all_assemblies)
    cols = min(3, n)
    rows = math.ceil(n / cols) if n > 0 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    x_common = np.linspace(0, 100, 101)
    
    for i, assembly in enumerate(all_assemblies):
        ax = axes[i]
        
        # Plot Original as blue dashed
        if assembly in global_orig_dict:
            ax.plot(x_common, global_orig_dict[assembly], color='#1f77b4', linestyle='--', linewidth=2.5, alpha=0.8, label='Original')
            
        # Plot New as orange solid
        if assembly in global_new_dict:
            ax.plot(x_common, global_new_dict[assembly], color='#ff7f0e', linestyle='-', linewidth=3.5, label='New')
            
        ax.set_title(assembly.upper(), fontsize=14, fontweight='bold')
        ax.set_xlabel("Insertion Progress (%)", fontsize=12)
        ax.set_ylabel("Avg Total Manipulability ($w_{move} + w_{hold}$)", fontsize=12)
        ax.grid(True, alpha=0.4)
        ax.legend()
        
    for i in range(len(all_assemblies), len(axes)):
        axes[i].set_visible(False)
        
    fig.suptitle("Average Insertion Kinematics per Assembly", fontsize=20, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.92 if rows > 1 else 0.85)
    
    save_path = os.path.join('results', test_name, 'insertion_plots', 'global_average_insertion.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated global summary plot: {save_path}")

def main(test_name):
    base_dir = os.path.join('logs', test_name)
    if not os.path.exists(base_dir): return
    
    # Dictionaries to catch the average total manipulability for each assembly
    global_orig_dict = {}
    global_new_dict = {}
    
    for assembly in [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]:
        generate_comparative_plots(base_dir, test_name, assembly, global_orig_dict, global_new_dict)
        
    # Generate the final master grid plot comparing all assemblies
    generate_global_summary_plot(test_name, global_orig_dict, global_new_dict)
    print(f"All comparative insertion plots safely routed to results/{test_name}/insertion_plots/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type=str)
    args = parser.parse_args()
    main(args.test_name)