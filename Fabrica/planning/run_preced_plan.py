import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

# project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# sys.path.append(project_base_dir)

# This points to the 'Fabrica' folder
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_base_dir)

# ADD THIS: This points one level higher to 'THESIS-CAD-BASED...', where 'matrix_code' lives
workspace_dir = os.path.abspath(os.path.join(project_base_dir, '..'))
sys.path.insert(0, workspace_dir)

import numpy as np
import networkx as nx
import pickle
from time import time
import json
import trimesh

from assets.load import load_part_ids, load_config
from planning.robot.geometry import load_part_meshes
from planning.sequence.feasibility_check import check_assemblable_parallel, check_path_collision, new_check_path_collision, check_ground_collision, new_check_ground_collision, CONTACT_EPS, generate_straight_path
from planning.robot.workcell import get_assembly_center
from utils.parallel import parallel_execute
from matrix_code.IM_Generation.functions import load_fabrica_assembly_from_folder, calculate_IM_matrices, get_freedom_score, get_free_directions, identify_ground_parts

def remove_redundant_edges(G):
    # Iterate over all pairs of nodes (u, v) in the graph
    for u in G.nodes():
        for v in list(G.successors(u)):  # Iterate over successors (outgoing edges)
            # For each successor v, check if u can reach v through any other path
            for intermediate in G.successors(u):  # Intermediate nodes
                if intermediate != v and nx.has_path(G, intermediate, v):
                    G.remove_edge(u, v)  # Remove redundant edge from u to v
                    break  # We only need to remove one redundant path, so we can stop
    return G


def assign_reachability_attributes(G):
    for node in G.nodes():
        G.nodes[node]['parts_before'] = list(set(nx.descendants(G, node)))  # Nodes the current node can reach
        G.nodes[node]['parts_after'] = list(set(nx.ancestors(G, node)))  # Nodes that can reach the current node
    return G


def compute_com(G, part_meshes_final, assembly_center):
    for node in G.nodes():
        part_mesh = part_meshes_final[node]
        G.nodes[node]['com'] = part_mesh.centroid + assembly_center
    return G


def compute_contact_points(G, part_meshes_final, config, assembly_center, contact_eps=CONTACT_EPS):
    # No more load_part_meshes() or load_config() reading from the hard drive!
    if config is not None and 'contact_eps' in config:
        contact_eps = config['contact_eps']  
    for edge in G.edges():
        part1, part2 = edge
        mesh1, mesh2 = part_meshes_final[part1], part_meshes_final[part2]
        proximity = trimesh.proximity.ProximityQuery(mesh1)
        closest_points, distances, _ = proximity.on_surface(mesh2.sample(int(100 * mesh2.area)))
        contact_points = closest_points[distances < contact_eps]
        G.edges[edge]['contact_points'] = contact_points + assembly_center
        # trimesh.Scene([mesh1, mesh2, trimesh.points.PointCloud(contact_points)]).show() # for debugging
    return G


def run_preced_plan(assembly_dir, log_dir, arm_type, num_proc=1, inner_num_proc=1, verbose=False, use_heuristic=False):
    '''
    Plan precedence tiers for assembly sequence.
    [
        { # tier 1
            part_11: {'action': action_11, 'path': path_11},
            part_12: {'action': action_12, 'path': path_12},
            ...
        },
        { # tier 2
            part_21: {'action': action_21, 'path': path_21},
            part_22: {'action': action_22, 'path': path_22},
            ...
        },
        ...
    ]
    '''
    asset_folder = os.path.join(project_base_dir, './assets')
    tiers = []
    parts_assembled = sorted(load_part_ids(assembly_dir)) # this sorts the parts in name order (which is why we implement heuristic)
    assembly_center = get_assembly_center(arm_type)
    config = load_config(assembly_dir) # load particular configuration for part assembly

    # ADDED: Pre-sort parts assembled based on freedom score
    master_part_ids = None
    directional_matrices_AABB = None
    directional_matrices_OBB = None
    assembly_manifest_AABB = None
    assembly_manifest_OBB = None
    
    t_start = time()
    if use_heuristic:
        master_part_ids = parts_assembled.copy()
        
        # Load ONLY the Global (AABB) manifest
        assembly_manifest_AABB = load_fabrica_assembly_from_folder(assembly_dir, master_part_ids, bounding_box_type="AABB")

        # Pre-calculate ONLY AABB interference matrices
        directional_matrices_AABB = calculate_IM_matrices(assembly_manifest_AABB)
        
        print(f'[run_preced_plan] AABB IM matrices calculated in {round(time() - t_start, 2)} seconds')

    
    while len(parts_assembled) > 1:
        tier = {}

        # Helper function to execute a group of parts via Fabrica's parallel workers
        # Upgraded helper function to execute a group of parts in chunks
        def evaluate_group(group_to_test, break_on_success=False):
            # If breaking early, test in chunks matching available CPU cores
            chunk_size = num_proc if break_on_success else len(group_to_test)
            
            for i in range(0, len(group_to_test), chunk_size):
                chunk = group_to_test[i:i+chunk_size]
                args, kwargs = [], []
                
                for part_move in chunk:
                    parts_fix = parts_assembled.copy()
                    parts_fix.remove(part_move)
                    args.append((asset_folder, assembly_dir, parts_fix, part_move))
                    kwargs.append(dict(
                        num_proc=inner_num_proc, pose=np.eye(4), save_sdf=True, 
                        return_path=True, optimize_path=True, debug=0, render=False,
                        directional_matrices=directional_matrices_AABB,  
                        master_part_ids=master_part_ids
                    ))
                
                if not args:
                    continue
                    
                found_success = False
                for (action, path), ret_arg, _ in parallel_execute(check_assemblable_parallel, args, kwargs, num_proc=num_proc, return_args=True, show_progress=verbose, desc='check_assemblable'):
                    if action is not None:
                        part_move = ret_arg[-1]
                        parts_assembled.remove(part_move)
                        tier[part_move] = {'action': action, 'path': path}
                        found_success = True
                
                # If we found at least one free part in this chunk, stop simulating the deeper parts!
                if break_on_success and found_success:
                    break

        # BRANCH: Heuristic ON vs OFF
        if use_heuristic:
            free_parts = []
            locked_parts = []
            
            ground_parts = identify_ground_parts(assembly_manifest_AABB)

            # --- STAGE 1: GLOBAL AABB LOOKUP ---
            for part_move in parts_assembled:
                free_dirs = get_free_directions(
                    part_id=part_move, 
                    current_assembly_ids=parts_assembled, 
                    part_ids_list=master_part_ids, 
                    matrices_dict=directional_matrices_AABB
                )
                
                if part_move in ground_parts and "-z" in free_dirs:
                    free_dirs.remove("-z")
                
                if len(free_dirs) > 0:
                    # Pull the exact physical vector from your manifest dictionary
                    action_vec = assembly_manifest_AABB[part_move]["extraction_vectors"][free_dirs[0]]
                    free_parts.append((part_move, action_vec))
                else:
                    locked_parts.append(part_move)

            # --- STAGE 2: LOCAL OBB LOOKUP ---
            still_locked_parts = []
            if len(free_parts) == 0:
                print(f'[run_preced_plan] No global free parts, checking local OBB directions for: {locked_parts}')
                
                # ---> LAZY COMPUTATION TRIGGER <---
                if directional_matrices_OBB is None:
                    print('[run_preced_plan] Cardinal lock detected! Lazy-loading OBB matrices...')
                    t_obb_start = time()
                    assembly_manifest_OBB = load_fabrica_assembly_from_folder(assembly_dir, master_part_ids, bounding_box_type="OBB")
                    directional_matrices_OBB = calculate_IM_matrices(assembly_manifest_OBB)
                    print(f'[run_preced_plan] OBB matrices lazily calculated in {round(time() - t_obb_start, 2)} seconds')
                
                for part_move in locked_parts:
                    free_dirs_obb = get_free_directions(
                        part_id=part_move, 
                        current_assembly_ids=parts_assembled, 
                        part_ids_list=master_part_ids, 
                        matrices_dict=directional_matrices_OBB
                    )
                    
                    safe_dir = None
                    for d in free_dirs_obb:
                        # Pull the EXACT diagonal OBB vector you already computed
                        action_vec = assembly_manifest_OBB[part_move]["extraction_vectors"][d]
                        
                        # Prevent downward extraction into the floor
                        if part_move in ground_parts and action_vec[2] < -0.1:
                            continue
                            
                        safe_dir = action_vec
                        break
                        
                    if safe_dir is not None:
                        free_parts.append((part_move, safe_dir))
                    else:
                        still_locked_parts.append(part_move)

            # ---> MAIN THREAD OVERRIDE: Instant Assignment <---
            if len(free_parts) > 0:
                for part_move, action_vec in free_parts:
                    parts_fix = parts_assembled.copy()
                    parts_fix.remove(part_move)
                    
                    path = generate_straight_path(
                        assembly_manifest=assembly_manifest_AABB, 
                        part_move_id=part_move, 
                        action_vec=action_vec, 
                        parts_fix=parts_fix
                    )
                    
                    parts_assembled.remove(part_move)
                    tier[part_move] = {'action': action_vec, 'path': path}
            
            # --- STAGE 3: PARALLEL WORKER FALLBACK ---
            if len(tier) == 0:
                print(f'[run_preced_plan] Absolute lock. Sorting and evaluating in parallel: {still_locked_parts}')
                
                # 1. Calculate Lock Severity (total AABB matrix collisions) and Volume
                lock_severities = {}
                volumes = {}
                for p in still_locked_parts:
                    part_idx = master_part_ids.index(p)
                    fixed_indices = [master_part_ids.index(pid) for pid in parts_assembled if pid != p]
                    
                    # Sum collisions across all 6 global directions
                    total_cols = sum(np.sum(directional_matrices_AABB[d][part_idx, fixed_indices]) for d in ["+x", "-x", "+y", "-y", "+z", "-z"])
                    lock_severities[p] = total_cols
                    
                    # Fetch volume directly from the fast AABB manifest
                    bounds = assembly_manifest_AABB[p]['part_mesh'].bounds
                    volumes[p] = np.prod(bounds[1] - bounds[0])
                    
                # 2. Sort primarily by least collisions (outer shell), secondarily by smallest volume
                still_locked_parts.sort(key=lambda p: (lock_severities[p], volumes[p]))
                
                # 3. Evaluate in chunks and BREAK EARLY to save time
                evaluate_group(still_locked_parts, break_on_success=True)
        else:
            # Original Fabrica baseline behavior
            print(f'[run_preced_plan] Evaluating all parts in parallel: {parts_assembled}')
            evaluate_group(parts_assembled)

        if len(tier) == 0:
            raise ValueError(f'[run_preced_plan] No parts in {parts_assembled} can be disassembled ({assembly_dir})')

        # Check if any part is touching the ground
        if len(tier) > 1 and len(parts_assembled) == 0:
            parts_on_ground = new_check_ground_collision(assembly_dir, list(tier.keys()))
            assert len(parts_on_ground) > 0, f'No parts in {list(tier.keys())} touches the ground'
            parts_floating = list(set(tier.keys()) - set(parts_on_ground))
            tier_floating, tier_on_ground = {part: tier[part] for part in parts_floating}, {part: tier[part] for part in parts_on_ground}
            if len(tier_floating) > 0:
                tiers.append(tier_floating) # disassemble floating parts before base parts
            tiers.append(tier_on_ground)
            if verbose:
                print(f'[run_preced_plan] precedence tier: {list(tier_floating.keys())}')
                print(f'[run_preced_plan] precedence tier: {list(tier_on_ground.keys())}')
        else:
            tiers.append(tier)
            if verbose:
                print(f'[run_preced_plan] precedence tier: {list(tier.keys())}')
    
    '''
    Convert precedence tiers to graph
    '''

    # Generate the directed graph based on tiers and centered on the table coordinates
    G = nx.DiGraph()
    for tier in tiers:
        for part, info in tier.items():
            # centered_path = np.array(info['path']) + np.concatenate([assembly_center, np.zeros(3)])
            # add zeros as the last 3 dimensions to make the PARTS path 6 dimensional
            centered_path = np.hstack([info['path'] + assembly_center, np.zeros((len(info['path']), 3))])
            G.add_node(part, action=info['action'], path=centered_path, parts_before=[], parts_after=[])
    for part in parts_assembled:
        G.add_node(part, action=None, path=None, parts_before=[], parts_after=[])
    
    parts_removed = []
    args, kwargs = [], []
    for tier in tiers:
        for part, info in tier.items():
            if len(parts_removed) == 0:
                continue
            # args.append((assembly_dir, part, parts_removed.copy(), info['path']))
            args.append((assembly_dir, part, parts_removed.copy(), np.hstack([info['path'], np.zeros((len(info['path']), 3))])))
            kwargs.append(dict(n_sample=5)) # check 5 samples for each path
        parts_removed.extend(list(tier.keys()))
    
    for parts_in_collision, ret_arg, _ in parallel_execute(new_check_path_collision, args, kwargs, num_proc=num_proc, return_args=True, show_progress=verbose, desc='check_path_collision'):
        part = ret_arg[1]
        if verbose and len(parts_in_collision) > 0:
            print(f'[run_preced_plan] {part} in collision with {parts_in_collision}')
        for part_collision in parts_in_collision:
            assert part_collision in G.nodes
            G.add_edge(part, part_collision)
    
    if len(parts_assembled) == 1: # add base part
        for part_other in tiers[-1]:
            G.add_edge(parts_assembled[0], part_other)

    # add precedence preference from config
    if config is not None and 'precedence' in config:
        for part_first, part_second in config['precedence']:
            assert part_first in G.nodes and part_second in G.nodes
            G.add_edge(part_first, part_second)

    G = remove_redundant_edges(G)
    G = assign_reachability_attributes(G)
    loaded_meshes = load_part_meshes(assembly_dir, transform='final', rename=False)
    G = compute_com(G, loaded_meshes, assembly_center)
    G = compute_contact_points(G, loaded_meshes, config, assembly_center)

    save_graph(G, log_dir)
    stats_path = os.path.join(log_dir, 'stats.json')
    with open(stats_path, 'w') as fp:
        json.dump({'preced_plan': {'time': round(time() - t_start, 2)}}, fp)


def draw_graph(G, save_path=None):
    from networkx.drawing.nx_pydot import graphviz_layout
    import matplotlib.pyplot as plt
    pos = graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos, arrows=True, with_labels=True)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def save_graph(G, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    graph_path = os.path.join(log_dir, 'precedence.pkl')
    image_path = os.path.join(log_dir, 'precedence.png')
    with open(graph_path, 'wb') as fp:
        pickle.dump(G, fp)
    draw_graph(G, save_path=image_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True, help='directory of assembly')
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--arm', type=str, default='panda', help='robot arm type')
    parser.add_argument('--num-proc', type=int, default=1, help='number of processes')
    parser.add_argument('--inner-num-proc', type=int, default=6, help='number of inner processes')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    args = parser.parse_args()

    run_preced_plan(args.assembly_dir, args.log_dir, args.arm, num_proc=args.num_proc, inner_num_proc=args.inner_num_proc, verbose=args.verbose, 
                    use_heuristic=True)
