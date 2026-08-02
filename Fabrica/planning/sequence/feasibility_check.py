import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import numpy as np
from itertools import combinations
from time import time
import trimesh
import concurrent.futures

from assets.load import load_assembly_all_transformed
from assets.transform import get_transform_from_path
from utils.renderer import SimRenderer
from utils.parallel import parallel_execute
from planning.sequence.physics_planner import MultiPartPathPlanner, MultiPartStabilityPlanner, MultiPartNoForceStabilityPlanner, get_contact_graph, CONTACT_EPS
from matrix_code.IM_Generation.functions import evaluate_pair_interference, load_fabrica_assembly_from_folder

def jit_check_single_direction(assembly_dir, parts_fix, part_move, action_vec, min_sep=None):
    """
    Evaluates a single direction using Hybrid AABB + Exact Math.
    Designed to run concurrently inside Redmax parallel workers.
    """
    # Only process pure cardinal directions
    if np.sum(np.abs(action_vec)) != 1.0:
        return False, None
        
    assembly = load_assembly_all_transformed(assembly_dir)
    
    part_z_mins = {name: data['mesh_final'].bounds[0][2] for name, data in assembly.items()}
    global_ground_z = min(part_z_mins.values())
    ground_parts = [name for name, z_min in part_z_mins.items() if abs(z_min - global_ground_z) <= 1e-3]
    if part_move in ground_parts and np.allclose(action_vec, [0, 0, -1]):
        return False, None

    if not parts_fix:
        path = generate_straight_path(assembly_dir, part_move, action_vec, parts_fix, min_sep, assembly=assembly)
        return True, path

    axis_idx = np.argmax(np.abs(action_vec))
    sign_val = action_vec[axis_idx]
    sign_str = 'pos' if sign_val > 0 else 'neg'
    axis_name = ['x', 'y', 'z'][axis_idx]

    mesh_a = assembly[part_move]['mesh']
    center_a = mesh_a.bounding_box.centroid
    to_origin_a = np.eye(4)
    to_origin_a[:3, 3] = -center_a
    part_a_data = {"part_mesh": mesh_a, "to_origin": to_origin_a}
    bounds_a = assembly[part_move]['mesh_final'].bounds

    for p_fix in parts_fix:
        bounds_b = assembly[p_fix]['mesh_final'].bounds
        
        # --- AABB GATE ---
        tol = 1e-3
        overlap_2d = True
        for i in range(3):
            if i == axis_idx: continue
            if bounds_a[1][i] <= bounds_b[0][i] + tol or bounds_a[0][i] >= bounds_b[1][i] - tol:
                overlap_2d = False
                break
                
        blocked = False
        if overlap_2d:
            if sign_str == 'pos':
                if bounds_b[1][axis_idx] > bounds_a[0][axis_idx] + tol:
                    # EXACT FALLBACK
                    mesh_b = assembly[p_fix]['mesh_final']
                    pos_val, _ = evaluate_pair_interference(
                        part_a_data, {"part_mesh": mesh_b}, axis_name, override_w_tol=0.01
                    )
                    if pos_val > 0: blocked = True
            else:
                if bounds_b[0][axis_idx] < bounds_a[1][axis_idx] - tol:
                    # EXACT FALLBACK
                    mesh_b = assembly[p_fix]['mesh_final']
                    _, neg_val = evaluate_pair_interference(
                        part_a_data, {"part_mesh": mesh_b}, axis_name, override_w_tol=0.01
                    )
                    if neg_val > 0: blocked = True

        if blocked:
            return False, None

    path = generate_straight_path(assembly_dir, part_move, action_vec, parts_fix, min_sep, assembly=assembly)
    return True, path

def generate_straight_path(assembly_manifest, part_move_id, action_vec, parts_fix=None, min_sep=None, n_steps=100):
    """
    Generates a 3D relative displacement path starting from [0, 0, 0].
    Now uses the blazing fast assembly_manifest.
    """
    # Safely pull the mesh regardless of which loader generated the dictionary
    mesh_a = assembly_manifest[part_move_id].get('part_mesh', assembly_manifest[part_move_id].get('mesh_final'))
    
    active_min_sep = min_sep if min_sep is not None else 0.5
    
    axis_idx = np.argmax(np.abs(action_vec))
    sign = action_vec[axis_idx] # +1 or -1
    my_bounds = mesh_a.bounds
    
    if not parts_fix:
        part_span = my_bounds[1][axis_idx] - my_bounds[0][axis_idx]
        total_distance = part_span + active_min_sep
    else:
        if sign > 0:
            max_fix_upper = max(assembly_manifest[pf].get('part_mesh', assembly_manifest[pf].get('mesh_final')).bounds[1][axis_idx] for pf in parts_fix)
            my_lower = my_bounds[0][axis_idx]
            total_distance = max(max_fix_upper - my_lower, 0) + active_min_sep
        else:
            min_fix_lower = min(assembly_manifest[pf].get('part_mesh', assembly_manifest[pf].get('mesh_final')).bounds[0][axis_idx] for pf in parts_fix)
            my_upper = my_bounds[1][axis_idx]
            total_distance = max(my_upper - min_fix_lower, 0) + active_min_sep

    # Redmax paths start at relative displacement [0, 0, 0]
    path = []
    for i in range(n_steps + 1):
        displacement = action_vec * (total_distance * (i / n_steps))
        path.append(displacement)
        
    return np.array(path)

def jit_check_cardinal_extractions(assembly_dir, parts_fix, part_move, min_sep=None, directional_matrices=None, master_part_ids=None):
    """
    Pure serial JIT. Relies on the outer multiprocessing pool from run_preced_plan.
    Uses O(1) matrix lookups if available, or falls back to fast Cython overlap logic.
    """
    part_ids = parts_fix + [part_move]
    assembly = load_fabrica_assembly_from_folder(assembly_dir, part_ids)
    
    part_z_mins = {name: data['part_mesh'].bounds[0][2] for name, data in assembly.items()}
    global_ground_z = min(part_z_mins.values())
    ground_parts = [name for name, z_min in part_z_mins.items() if abs(z_min - global_ground_z) <= 1e-3]
    is_on_ground = part_move in ground_parts

    if not parts_fix:
        action = np.array([0, 0, 1])
        path = generate_straight_path(
            assembly_manifest=assembly, 
            part_move_id=part_move, 
            action_vec=action, 
            parts_fix=parts_fix, 
            min_sep=min_sep
        )
        return action, path

    # Include the matrix string key for easy lookups
    cardinal_tests = [
        (np.array([0, 0, 1]), 'z', 'pos', 2, '+z'),   
        (np.array([0, 0, -1]), 'z', 'neg', 2, '-z'),  
        (np.array([1, 0, 0]), 'x', 'pos', 0, '+x'),   
        (np.array([-1, 0, 0]), 'x', 'neg', 0, '-x'),  
        (np.array([0, 1, 0]), 'y', 'pos', 1, '+y'),   
        (np.array([0, -1, 0]), 'y', 'neg', 1, '-y')   
    ]

    use_matrices = directional_matrices is not None and master_part_ids is not None
    
    if use_matrices:
        move_idx = master_part_ids.index(part_move)
    else:
        part_a_data = assembly[part_move]
        bounds_a = part_a_data['part_mesh'].bounds

    for action, axis_name, sign_str, axis_idx, dir_key in cardinal_tests:
        if is_on_ground and np.allclose(action, [0, 0, -1]):
            continue

        direction_clear = True
        for p_fix in parts_fix:
            blocked = False
            
            # ---> O(1) MATRIX LOOKUP ROUTE <---
            if use_matrices:
                fix_idx = master_part_ids.index(p_fix)
                if directional_matrices[dir_key][move_idx, fix_idx] > 0:
                    blocked = True
                    
            # ---> DYNAMIC CYTHON FALLBACK ROUTE <---
            else:
                part_b_data = assembly[p_fix]
                bounds_b = part_b_data['part_mesh'].bounds
                
                tol = 1e-3
                overlap_2d = True
                for i in range(3):
                    if i == axis_idx: continue
                    if bounds_a[1][i] <= bounds_b[0][i] + tol or bounds_a[0][i] >= bounds_b[1][i] - tol:
                        overlap_2d = False
                        break
                        
                if overlap_2d:
                    pos_val, neg_val = evaluate_pair_interference(
                        part_a_data, part_b_data, axis_name, 
                        override_w_tol=0.01, abort_threshold=75000,
                        use_cython=True, use_parallel_narrow=False
                    )
                    
                    if pos_val == -999:
                        blocked = True
                        direction_clear = False 
                        break

                    if sign_str == 'pos':
                        if pos_val > 0: blocked = True
                    else:
                        if neg_val > 0: blocked = True

            if blocked:
                direction_clear = False
                break

        if direction_clear:
            path = generate_straight_path(
                assembly_manifest=assembly, 
                part_move_id=part_move, 
                action_vec=action, 
                parts_fix=parts_fix, 
                min_sep=min_sep
            )
            return action, path

    return None, None


def get_R3_actions():
    actions = [
        np.array([0, 0, 1]), # +Z
        np.array([0, 0, -1]), # -Z
        np.array([1, 0, 0]), # +X
        np.array([-1, 0, 0]), # -X
        np.array([0, 1, 0]), # +Y
        np.array([0, -1, 0]), # -Y   
    ]
    return actions


def check_assemblable(asset_folder, assembly_dir, parts_fix, part_move, pose=None, save_sdf=False, debug=0, render=False, return_path=False, optimize_path=False, min_sep=None, adaptive_sample=False):
    '''
    Check if certain parts are disassemblable
    '''
    planner = MultiPartPathPlanner(asset_folder, assembly_dir, parts_fix, part_move, pose=pose, save_sdf=save_sdf, adaptive_sample=adaptive_sample)

    actions = get_R3_actions()
    best_action = None
    best_path = None
    best_path_len = np.inf
    for action in actions:
        success, path = planner.check_success(action, return_path=True, min_sep=min_sep, max_path_len=best_path_len)
        if debug > 0:
            print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action: {action}, path_len: {len(path)}')
            if render:
                SimRenderer().replay(planner.sim)
        if success:
            if len(path) < best_path_len:
                best_path_len = len(path)
                best_path = path
                best_action = action

    if best_path is not None:
        best_path = np.array(best_path)
        if optimize_path: # optimize action based on the path found
            best_dirs = best_path[1:, :3] - best_path[:-1, :3]
            best_dirs = best_dirs[np.linalg.norm(best_dirs, axis=1) > 1e-6]
            opt_action = np.median(best_dirs / np.linalg.norm(best_dirs, axis=1)[:, None], axis=0)
            opt_action = opt_action / np.linalg.norm(opt_action)
            success, opt_path = planner.check_success(opt_action, return_path=True, min_sep=min_sep)
            if debug > 0:
                print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action (optimized): {opt_action}, path_len (optimized): {len(opt_path)}')
                if render:
                    SimRenderer().replay(planner.sim)
            if success:
                best_path_len = len(opt_path)
                best_path = opt_path
                best_action = opt_action
        best_path = np.array(best_path)

    if return_path:
        return best_action, best_path
    else:
        return best_action


def _check_assemblable_worker(asset_folder, assembly_dir, parts_fix, part_move, pose, save_sdf, optimize_path, min_sep, adaptive_sample, action, debug, render):
    '''
    Worker process for check_assemblable_parallel
    '''
    # check_success is used to check if the action is feasible ONLY FOR THE MOVING PART, NOT INCLUDING ROBOT ARM YET
    planner = MultiPartPathPlanner(asset_folder, assembly_dir, parts_fix, part_move, pose=pose, save_sdf=save_sdf, adaptive_sample=adaptive_sample)
    success, path = planner.check_success(action, return_path=True, min_sep=min_sep)

    if debug > 0:
        print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action: {action}, path_len: {len(path)}')
        if render:
            SimRenderer().replay(planner.sim)

    # HERE WE OPTIMIZE THE ACTUAL ACTION BASED ON THE PATH SINCE ACTION MAY NOT BE ALIGNED WITH IT
    # HERE SINCE WE HAVE A PATH DIRECTION WE CAN IMPLEMENT THE 2D CHECKS WITH MY CODE BUT ONLY ON THIS DIRECTION
    if success:
        assert path is not None
        path = np.array(path)
        if optimize_path: # optimize action based on the path found
            dirs = path[1:, :3] - path[:-1, :3]
            dirs = dirs[np.linalg.norm(dirs, axis=1) > 1e-6]
            opt_action = np.median(dirs / np.linalg.norm(dirs, axis=1)[:, None], axis=0)
            opt_action = opt_action / np.linalg.norm(opt_action)
            success, opt_path = planner.check_success(opt_action, return_path=True, min_sep=min_sep)
            if debug > 0:
                print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action (optimized): {opt_action}, path_len (optimized): {len(opt_path)}')
                if render:
                    SimRenderer().replay(planner.sim)
            if success:
                path = opt_path
                action = opt_action
        path = np.array(path)

    return success, path, action


def check_assemblable_parallel(asset_folder, assembly_dir, parts_fix, part_move, num_proc, pose=None, save_sdf=False, debug=0, render=False, return_path=False, optimize_path=False, min_sep=None, adaptive_sample=False, directional_matrices=None, master_part_ids=None):
    '''
    Parallel version of check_assemblable
    '''

    # --- JIT ANALYTICAL PRE-FILTER ---
    if directional_matrices is None:
        action, path = jit_check_cardinal_extractions(assembly_dir, parts_fix, part_move, min_sep, directional_matrices, master_part_ids)
        if action is not None:
            if 1:
                print(f'[JIT Pre-Filter] Successfully bypassed Redmax simulation for {part_move} along {action}!')
            if return_path:
                return action, path
            else:
                return action
    # ---------------------------------

    # HERE WE ARE TESTING ALL THE POSSIBLE ACTIONS IN R3 WHICH IS NOT EFFICIENT
    # HOWEVER WE CANT FILTER WITH MY CARDINAL 2D MATRICES SINCE AN ACTION MAY NOT BE ALIGNED WITH IT
    # AND PART MAY BE EXTRACTED ANYWAYS 
    # DONT IMPLEMENT MATRICES ON CHECK ASSEMBLABLE, ITS BETTER TO CHECK ONCE WE OPTIMIZE THE ACTION BASED ON THE PATH FOUND
    actions = get_R3_actions()
    if num_proc < len(actions):
        return check_assemblable(asset_folder, assembly_dir, parts_fix, part_move, pose=pose, save_sdf=save_sdf, debug=debug, render=render, return_path=return_path, optimize_path=optimize_path, min_sep=min_sep, adaptive_sample=adaptive_sample)
    
    best_action = None
    best_path = None
    best_path_len = np.inf

    worker_args = []
    for action in actions:
        worker_args.append((asset_folder, assembly_dir, parts_fix, part_move, pose, save_sdf, optimize_path, min_sep, adaptive_sample, action, 0, False))

    for (success, path, action) in parallel_execute(_check_assemblable_worker, worker_args, num_proc=num_proc, terminate_func=None, show_progress=False):
        if debug > 0:
            print(f'[check_assemblable] success: {success}, parts_fix: {parts_fix}, part_move: {part_move}, action: {action}, path_len: {len(path)}')
        if success and len(path) < best_path_len:
            best_path = path
            best_path_len = len(path)
            best_action = action

    if return_path:
        return best_action, best_path
    else:
        return best_action


def check_all_connection_assemblable(asset_folder, assembly_dir, parts=None, contact_eps=CONTACT_EPS, save_sdf=False, num_proc=1, debug=0, render=False):
    '''
    Check if all connected pairs of parts are disassemblable
    '''
    G = get_contact_graph(asset_folder, assembly_dir, parts, contact_eps=contact_eps, save_sdf=save_sdf)

    worker_args = []
    for pair in G.edges:
        part_a, part_b = pair
        worker_args.append([asset_folder, assembly_dir, [part_a], part_b, None, save_sdf, debug, render])

    failures = []
    for action, args in parallel_execute(check_assemblable, worker_args, num_proc=num_proc, show_progress=debug > 0, desc='check_all_connection_assemblable', return_args=True):
        success = action is not None
        part_fix, part_move = args[2][0], args[3]
        if debug > 0:
            print(f'[check_all_connection_assemblable] success: {success}, part_fix: {part_fix}, part_move: {part_move}, action: {action}')
        if not success:
            failures.append((part_fix, part_move))

    all_success = len(failures) == 0
    return all_success, failures


def check_given_connection_assemblable(asset_folder, assembly_dir, part_pairs, bidirection=False, save_sdf=False, num_proc=1, debug=0, render=False):
    '''
    Check if given connected pairs of parts are disassemblable
    '''
    worker_args = []
    for pair in part_pairs:
        part_a, part_b = pair
        worker_args.append([asset_folder, assembly_dir, [part_a], part_b, None, save_sdf, debug, render])
        if bidirection:
            worker_args.append([asset_folder, assembly_dir, [part_b], part_a, None, save_sdf, debug, render])

    failures = []
    for action, args in parallel_execute(check_assemblable, worker_args, num_proc=num_proc, show_progress=debug > 0, desc='check_given_connection_assemblable', return_args=True):
        success = action is not None
        part_fix, part_move = args[2][0], args[3]
        if debug > 0:
            print(f'[check_given_connection_assemblable] success: {success}, part_fix: {part_fix}, part_move: {part_move}, action: {action}')
        if not success:
            failures.append((part_fix, part_move))

    all_success = len(failures) == 0
    return all_success, failures


def check_path_collision(assembly_dir, part_move, parts_other, path, n_sample=None):
    '''
    Check if path of part_move collides with parts_other
    '''
    if len(parts_other) == 0: return []
    assembly = load_assembly_all_transformed(assembly_dir)
    col_manager_move = trimesh.collision.CollisionManager()
    col_manager_move.add_object(part_move, assembly[part_move]['mesh'])
    col_manager_other = trimesh.collision.CollisionManager()
    for part_id, part in assembly.items():
        if part_id in parts_other:
            col_manager_other.add_object(part_id, part['mesh_final'])
    parts_in_collision = []
    transforms = get_transform_from_path(path, n_sample=n_sample)
    for transform in transforms:
        col_manager_move.set_transform(part_move, transform)
        in_collision, col_pairs = col_manager_other.in_collision_other(col_manager_move, return_names=True)
        if in_collision:
            for col_pair in col_pairs:
                if col_pair[0] not in parts_in_collision:
                    parts_in_collision.append(col_pair[0])
    return parts_in_collision

def new_check_path_collision(assembly_dir, part_move, parts_other, path, n_sample=None):
    if len(parts_other) == 0: return []
    
    # ---> USE THE FAST CACHE LOADER <---
    part_ids = parts_other + [part_move]
    assembly = load_fabrica_assembly_from_folder(assembly_dir, part_ids)
    
    path_arr = np.array(path)
    pts = path_arr[:, :3]
    
    direction = pts[-1] - pts[0]
    dir_norm = np.linalg.norm(direction)
    
    is_cardinal_straight = False
    axis_name = None
    sign_str = None
    
    if dir_norm > 1e-6:
        action_vec = direction / dir_norm
        segments = pts[1:] - pts[:-1]
        seg_norms = np.linalg.norm(segments, axis=1, keepdims=True)
        valid_mask = (seg_norms > 1e-6).flatten()
        
        if np.any(valid_mask):
            seg_dirs = segments[valid_mask] / seg_norms[valid_mask]
            dots = np.dot(seg_dirs, action_vec)
            is_straight = np.allclose(dots, 1.0, atol=1e-3)
            
            if is_straight:
                abs_vec = np.abs(action_vec)
                if np.max(abs_vec) > 0.999: 
                    axis_idx = np.argmax(abs_vec)
                    axis_name = ['x', 'y', 'z'][axis_idx]
                    sign_str = 'pos' if action_vec[axis_idx] > 0 else 'neg'
                    is_cardinal_straight = True

    suspects_for_fcl = []
    parts_in_collision = []
    
    if is_cardinal_straight:
        part_a_data = assembly[part_move]
    
    for part_other in parts_other:
        if is_cardinal_straight:
            part_b_data = assembly[part_other]
            
            # ---> NATIVE CYTHON JIT EXECUTION <---
            pos_val, neg_val = evaluate_pair_interference(
                part_a_data, part_b_data, axis_name, 
                override_w_tol=0.01, abort_threshold=None,
                use_cython=True, use_parallel_narrow=False
            )
            
            if pos_val == -999:
                suspects_for_fcl.append(part_other)
            else:
                val = pos_val if sign_str == 'pos' else neg_val
                if val > 0:
                    parts_in_collision.append(part_other)
        else:
            suspects_for_fcl.append(part_other)

    if suspects_for_fcl:
        col_manager_move = trimesh.collision.CollisionManager()
        col_manager_move.add_object(part_move, assembly[part_move]['part_mesh']) 
        
        col_manager_other = trimesh.collision.CollisionManager()
        for part_id in suspects_for_fcl:
            col_manager_other.add_object(part_id, assembly[part_id]['part_mesh'])
            
        transforms = get_transform_from_path(path, n_sample=n_sample)
        for transform in transforms:
            col_manager_move.set_transform(part_move, transform)
            in_collision, col_pairs = col_manager_other.in_collision_other(col_manager_move, return_names=True)
            if in_collision:
                for col_pair in col_pairs:
                    if col_pair[0] not in parts_in_collision:
                        parts_in_collision.append(col_pair[0])
                        
    return parts_in_collision

def check_ground_collision(assembly_dir, parts):
    '''
    Check if parts collide with ground
    '''
    assembly = load_assembly_all_transformed(assembly_dir)
    col_manager = trimesh.collision.CollisionManager()
    for part_id in parts:
        col_manager.add_object(part_id, assembly[part_id]['mesh_final'])
    ground_mesh = trimesh.creation.box((1000, 1000, 0.2)) # NOTE: 0.1cm will be detected
    col_manager_ground = trimesh.collision.CollisionManager()
    col_manager_ground.add_object('ground', ground_mesh)
    in_collision, col_pairs = col_manager.in_collision_other(col_manager_ground, return_names=True)
    parts_in_collision = [col_pair[0] for col_pair in col_pairs]
    return parts_in_collision

def new_check_ground_collision(assembly_dir, parts):
    '''
    Check if parts collide with ground mathematically.
    Bypasses the heavy CollisionManager entirely.
    '''
    assembly = load_assembly_all_transformed(assembly_dir)
    
    parts_in_collision = []
    
    # The original ground box was Z-thickness 0.2, centered at origin.
    # Therefore, its top face is at Z = 0.1. 
    # Any part crossing Z = 0.1 + microscopic noise is touching the ground.
    ground_z_threshold = 0.1 + 1e-5 

    for part_id in parts:
        part_z_min = assembly[part_id]['mesh_final'].bounds[0][2]
        
        if part_z_min <= ground_z_threshold:
            parts_in_collision.append(part_id)
            
    return parts_in_collision


def check_stable_noforce(asset_folder, assembly_dir, parts, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False):
    '''
    Check if stable without any external force
    '''
    planner = MultiPartNoForceStabilityPlanner(asset_folder, assembly_dir, parts, save_sdf=save_sdf, allow_gap=allow_gap)
    
    success, G = planner.check_success(timeout=timeout)
    if debug > 0:
        print(f'[check_stable_noforce] success: {success}')
        if render:
            SimRenderer().replay(planner.sim)

    return success, G


def check_stable(asset_folder, assembly_dir, parts_fix, parts_move, pose=None, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False):
    '''
    Check if gravitationally stable for a given fixed part
    '''
    planner = MultiPartStabilityPlanner(asset_folder, assembly_dir, parts_fix, parts_move, pose=pose, save_sdf=save_sdf, allow_gap=allow_gap)

    success, parts_fall = planner.check_success(timeout=timeout)
    if debug > 0:
        print(f'[check_stable] success: {success}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')
        if render:
            SimRenderer().replay(planner.sim)

    return success, parts_fall


def get_stable_plan_1pose_serial(asset_folder, assembly_dir, parts, base_part, pose, max_fix=None, save_sdf=False, timeout=None, allow_gap=False, debug=0, render=False, return_count=False):
    '''
    Get all gravitationally stable plans given 1 pose through serial greedy search
    '''
    t_start = time()
    count = 0

    max_fix = len(parts) if max_fix is None else min(max_fix, len(parts))
    parts_fix = [] if base_part is None else [base_part]
    
    while True:

        parts_move = parts.copy()
        for part_fix in parts_fix:
            parts_move.remove(part_fix)

        if timeout is not None:
            timeout -= (time() - t_start)
            if timeout < 0:
                if return_count:
                    return None, count
                else:
                    return None
            t_start = time()

        success, parts_fall = check_stable(asset_folder, assembly_dir, parts_fix, parts_move, pose, save_sdf, timeout, allow_gap, debug, render)
        count += 1

        if debug > 0:
            print(f'[get_stable_plan_1pose_serial] success: {success}, n_fix: {len(parts_fix)}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')

        if success:
            break
        else:
            if parts_fall is None:
                if return_count:
                    return None, count # timeout
                else:
                    return None
            parts_fix.extend(parts_fall)
        
        if len(parts_fix) > max_fix:
            if return_count:
                return None, count # failed
            else:
                return None

    if base_part is not None:
        parts_fix.remove(base_part)

    if return_count:
        return parts_fix, count
    else:
        return parts_fix


def get_stable_plan_1pose_parallel(asset_folder, assembly_dir, parts, base_part, pose=None, max_fix=None, save_sdf=False, timeout=None, allow_gap=False, num_proc=1, debug=0, render=False):
    '''
    Get all gravitationally stable plans given 1 pose through parallel greedy search
    '''
    t_start = time()

    max_fix = len(parts) if max_fix is None else min(max_fix, len(parts))

    if pose is not None:
        parts_fix = [] if base_part is None else [base_part]
        success, parts_fall = check_stable(asset_folder, assembly_dir, parts_fix, parts, pose, save_sdf, timeout, allow_gap, debug, render) # check if stable without any grippers
        if debug > 0:
            print(f'[get_stable_plan_1pose_parallel] success: {success}, n_fix: 0, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts}')
        if success:
            return []
        else:
            if parts_fall is None:
                return None # timeout

    if base_part is None:
        parts_fix_list = [[part_fix] for part_fix in parts]
    else:
        parts_fix_list = [[part_fix, base_part] for part_fix in parts if part_fix != base_part]
    
    while True:
        success_any = False

        if timeout is not None:
            timeout -= (time() - t_start)
            if timeout < 0:
                return None
            t_start = time()

        worker_args = []
        for parts_fix in parts_fix_list:
            if len(parts_fix) > max_fix: continue
            parts_move = parts.copy()
            for part_fix in parts_fix:
                parts_move.remove(part_fix)
            worker_args.append([asset_folder, assembly_dir, parts_fix, parts_move, pose, save_sdf, timeout, allow_gap, debug, render])

        if len(worker_args) == 0:
            return None # failed

        for (success, parts_fall), args in parallel_execute(check_stable, worker_args, num_proc=num_proc, show_progress=debug > 0, desc='get_stable_plan_1pose_parallel', return_args=True):
            parts_fix, parts_move = args[2], args[3]
            if debug > 0:
                print(f'[get_stable_plan_1pose_parallel] success: {success}, n_fix: {len(parts_fix)}, parts_fall: {parts_fall}, parts_fix: {parts_fix}, parts_move: {parts_move}')
            if success:
                success_any = True
            else:
                if parts_fall is None:
                    return None # timeout
                index = parts_fix_list.index(parts_fix)
                parts_fix_list[index].extend(parts_fall)
            if timeout is not None and time() - t_start > timeout:
                return None

        if success_any:
            break

    parts_fix_list = [parts_fix for parts_fix in parts_fix_list if len(parts_fix) <= max_fix]
    for parts_fix in parts_fix_list:
        if base_part is not None:
            parts_fix.remove(base_part)
    parts_fix_list = sorted(parts_fix_list, key=lambda x: len(x))
    return parts_fix_list
