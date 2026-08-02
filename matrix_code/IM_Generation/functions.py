import trimesh
import pyvista as pv
import numpy as np
import networkx as nx
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, workspace_dir)

import pandas as pd
import time
from pathlib import Path
from .classes import PseudoFace
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, Point
from itertools import product, permutations
import fast_simplification
from Fabrica.utils.parallel import fast_parallel_execute
from .narrow_phase_c import get_intersecting_pairs_c, fast_any_intersection_c, evaluate_deep_narrow_phase_c
import multiprocessing

class FastPart:
    """A lightweight proxy class to replace heavy Trimesh objects in the inner loop."""
    def __init__(self, triangles, face_normals):
        self.triangles = triangles
        self.face_normals = face_normals
        # Calculate 3D bounding box identically to Trimesh bounds[0], bounds[1]
        self.bounds = np.array([triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))])

def transform_mesh_data(triangles, normals, matrix):
    """Pure NumPy vectorization to apply a 4x4 transform instantly."""
    # Transform Triangles
    shape = triangles.shape
    reshaped = triangles.reshape(-1, 3)
    ones = np.ones((reshaped.shape[0], 1))
    homogenous = np.hstack([reshaped, ones])
    transformed_tris = (homogenous @ matrix.T)[:, :3].reshape(shape)
    
    # Transform Normals (Rotation Only)
    rot_matrix = matrix[:3, :3]
    transformed_normals = normals @ rot_matrix.T
    
    return transformed_tris, transformed_normals

ANGLE_NORMAL_TOL = 0.2
DISTANCE_TOL = 2e-4
W_TOL = 2e-4
FLUSH_TOL = 0.04
# W_TOL = 0.04

# ----------------------------------------------------- MAIN FUNCTIONS ----------------------------------------------
## AABB overlap test functions

def check_2d_aabb_overlap(bounds_a, bounds_b, extraction_axis, buffer=DISTANCE_TOL):
    overlap_region = {}
    # Note: This check considers the bounding boxes as inputted here, so the orientation depends
    # on how the bounds are defined before calling the function. 
    # For this check it's based on the oriented bounding boxes of part_a, and part_b is transformed
    # to match that orientation. As such it checks the extraction of part_a along its oriented bounding box 
    # axes. If you want to check the extraction along the world axes, you would need to ensure that the bounds
    # are defined in the world coordinate system before calling this function.
    """
    Squashes 3D bounding boxes onto a 2D plane based on the extraction axis
    and checks if the 2D rectangles overlap.
    extraction_axis: 0 for X, 1 for Y, 2 for Z
    """

    # Figure out which two axes form our 2D "shadow" plane
    # If we extract in Z (2), our 2D plane uses X (0) and Y (1).
    overlap_region = {}
    axis_idx = {"x": 0, "y": 1, "z": 2}
    all_axes = [0, 1, 2]
    all_axes.remove(axis_idx[extraction_axis])
    u_axis = all_axes[0]
    v_axis = all_axes[1]
    w_axis = axis_idx[extraction_axis]
    
    # Extract the Min and Max for the U axis (e.g., the X axis)
    # bounds[0] is Min, bounds[1] is Max
    a_min_u, a_max_u = bounds_a[0][u_axis], bounds_a[1][u_axis]
    b_min_u, b_max_u = bounds_b[0][u_axis], bounds_b[1][u_axis]
    
    # Extract the Min and Max for the V axis (e.g., the Y axis)
    a_min_v, a_max_v = bounds_a[0][v_axis], bounds_a[1][v_axis]
    b_min_v, b_max_v = bounds_b[0][v_axis], bounds_b[1][v_axis]
    
    # Calculate the exact boundaries of the overlap region (The SR)
    overlap_min_u = max(a_min_u, b_min_u)
    overlap_max_u = min(a_max_u, b_max_u)

    overlap_min_v = max(a_min_v, b_min_v)
    overlap_max_v = min(a_max_v, b_max_v)

    #buffer = DISTANCE_TOL
    overlap_region['overlap_u'] = (overlap_min_u - buffer, overlap_max_u + buffer)
    overlap_region['overlap_v'] = (overlap_min_v - buffer, overlap_max_v + buffer)

    if not ((overlap_min_u <= overlap_max_u) and (overlap_min_v <= overlap_max_v)):
          # No overlap
        return (overlap_region, 0)
    
    # Check COAABB overlap
    a_lims = [(a_min_u, a_max_u), (a_min_v, a_max_v)]
    b_lims = [(b_min_u, b_max_u), (b_min_v, b_max_v)]
    coaabb_overlap = check_COAABB_overlap(a_lims, b_lims)

    # Case 2: AABBs overlap but COAABBs do not (We need to check PFs) --> Return -2
    if not coaabb_overlap:
        return (overlap_region, -2)

    a_min_w, a_max_w = bounds_a[0][w_axis], bounds_a[1][w_axis]
    b_min_w, b_max_w = bounds_b[0][w_axis], bounds_b[1][w_axis]

    overlap_result = None
    if a_min_w > b_max_w + buffer:
         overlap_result = -2 
    elif b_min_w > a_max_w + buffer:
        overlap_result = -2   
    else:
        overlap_result = -2   

    return (overlap_region, overlap_result)

    # Note: The return values are as follows:
    #  0: No overlap at all (AABBs don't even touch)
    # -2: AABBs overlap but COAABBs do not (We need to check PFs)
    # -1: A cannot be extracted in the negative extraction direction without colliding with B
    #  1: A cannot be extracted in the positive extraction direction without colliding with B, 
    #    but can be extracted in the negative direction
    #  2: A cannot be extracted in either direction without colliding with B

def check_COAABB_overlap(a_lims, b_lims, epsilon=0.05):
    # ENGINEERING CAP: Maximum shrink in mm. 
    # Prevents massive parts from shrinking so much that they sever shallow joints!
    max_shrink = 0.5 

    # --- PART A CORE CALCULATION ---
    lua = a_lims[0][1] - a_lims[0][0]
    lva = a_lims[1][1] - a_lims[1][0]
    
    shrink_ua = min(epsilon * lua, max_shrink)
    shrink_va = min(epsilon * lva, max_shrink)
    
    Ua_min_core = a_lims[0][0] + shrink_ua
    Ua_max_core = a_lims[0][1] - shrink_ua
    Va_min_core = a_lims[1][0] + shrink_va
    Va_max_core = a_lims[1][1] - shrink_va

    # --- PART B CORE CALCULATION ---
    lub = b_lims[0][1] - b_lims[0][0]
    lvb = b_lims[1][1] - b_lims[1][0]
    
    shrink_ub = min(epsilon * lub, max_shrink)
    shrink_vb = min(epsilon * lvb, max_shrink)
    
    Ub_min_core = b_lims[0][0] + shrink_ub
    Ub_max_core = b_lims[0][1] - shrink_ub
    Vb_min_core = b_lims[1][0] + shrink_vb
    Vb_max_core = b_lims[1][1] - shrink_vb

    # --- TRUE 2D INTERSECTION MATH ---
    overlap_u = (Ua_min_core <= Ub_max_core) and (Ua_max_core >= Ub_min_core)
    overlap_v = (Va_min_core <= Vb_max_core) and (Va_max_core >= Vb_min_core)
    
    return overlap_u and overlap_v


## Pseudo Face overlap test functions 
def create_PFs(part: trimesh.Trimesh, extraction_axis: str, adj, tolerance = ANGLE_NORMAL_TOL):
    axis_idx = {"x": 0, "y": 1, "z": 2}
    w_idx = axis_idx[extraction_axis]

    # Gives unit normal vector that goes in extraction direction
    normals_w = part.face_normals[:, w_idx]

    # Strict Directional Separation 
    pos_mask = normals_w > tolerance
    neg_mask = normals_w < -tolerance

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    # ---> NOW USING THE INSTANT CACHED ADJACENCY <---
    pos_pairs = adj[np.isin(adj[:, 0], pos_indices) & np.isin(adj[:, 1], pos_indices)]
    neg_pairs = adj[np.isin(adj[:, 0], neg_indices) & np.isin(adj[:, 1], neg_indices)]

    # Build the Positive Graph
    G_pos = nx.Graph()
    G_pos.add_nodes_from(pos_indices)
    G_pos.add_edges_from(pos_pairs)

    # Build the Negative Graph
    G_neg = nx.Graph()
    G_neg.add_nodes_from(neg_indices)
    G_neg.add_edges_from(neg_pairs)

    # Combine the isolated components from both graphs
    components = list(nx.connected_components(G_pos)) + list(nx.connected_components(G_neg))

    # Convert to a list to ensure compatibility with your PseudoFace class
    return [PseudoFace(part, list(c), extraction_axis) for c in components if len(c) > 0]


def check_PF_overlap_original(pf_a: PseudoFace, pf_b: PseudoFace, flush_tol = FLUSH_TOL, angle_tol = ANGLE_NORMAL_TOL):
    result = [0, 0] 
    w_idx = pf_a.extraction_axis

    a_min_u, a_min_v = pf_a.triangles_2d.min(axis=(0,1))
    a_max_u, a_max_v = pf_a.triangles_2d.max(axis=(0,1))
    a_min_w = pf_a.triangles_3d[:, :, w_idx].min()
    a_max_w = pf_a.triangles_3d[:, :, w_idx].max()

    b_min_u, b_min_v = pf_b.triangles_2d.min(axis=(0,1))
    b_max_u, b_max_v = pf_b.triangles_2d.max(axis=(0,1))
    b_min_w = pf_b.triangles_3d[:, :, w_idx].min()
    b_max_w = pf_b.triangles_3d[:, :, w_idx].max()

    overlap_min_u = max(a_min_u, b_min_u)
    overlap_max_u = min(a_max_u, b_max_u)
    overlap_min_v = max(a_min_v, b_min_v)
    overlap_max_v = min(a_max_v, b_max_v)

    if not ((overlap_min_u <= overlap_max_u) and (overlap_min_v <= overlap_max_v)):
        return [0, 0] 
    
    a_lims = [(a_min_u, a_max_u), (a_min_v, a_max_v)]
    b_lims = [(b_min_u, b_max_u), (b_min_v, b_max_v)]
    coaabb_overlap = check_COAABB_overlap(a_lims, b_lims)

    if not coaabb_overlap:
        return [0, 0] 

    # ---> EXACT SHAPELY OVERLAP FILTER <---
    min_b_all = pf_b.triangles_2d.min(axis=1)
    max_b_all = pf_b.triangles_2d.max(axis=1)

    actual_overlap = False
    for tri_a in pf_a.triangles_2d:
        min_a = tri_a.min(axis=0)
        max_a = tri_a.max(axis=0)
        
        overlap_u = (min_a[0] <= max_b_all[:, 0]) & (max_a[0] >= min_b_all[:, 0])
        overlap_v = (min_a[1] <= max_b_all[:, 1]) & (max_a[1] >= min_b_all[:, 1])

        candidate_b_indices = np.where(overlap_u & overlap_v)[0]
        
        if len(candidate_b_indices) > 0:
            # ---> FIX: ADD 1e-4 BUFFER TO PREVENT MESH MISALIGNMENT SKIPS <---
            # poly_a = Polygon(tri_a).buffer(1e-4)
            poly_a = Polygon(tri_a)
            for idx_b in candidate_b_indices:
                # poly_b = Polygon(pf_b.triangles_2d[idx_b]).buffer(1e-4)
                poly_b = Polygon(pf_b.triangles_2d[idx_b])
                if poly_a.intersects(poly_b):
                    actual_overlap = True
                    break
        if actual_overlap:
            break

    if not actual_overlap:
        return [0, 0] 

    # -------------------------------------------------------------
    # THE KINEMATIC DIRECTIONAL CHECK
    # -------------------------------------------------------------
    normal_a_w = pf_a.part.face_normals[pf_a.face_indices[0]][w_idx]
    normal_b_w = pf_b.part.face_normals[pf_b.face_indices[0]][w_idx]

    # FLUSH TOLERANCE: 
    # DISTANCE_TOL (0.0002) is too strict for tessellated CAD. 
    # We use 0.05mm to absorb the natural mesh "criss-crossing" of flush parts.
    #flush_tol = FLUSH_TOL

    # 1. Deep Volume Overlap 
    # If they overlap by MORE than the flush tolerance, they are truly embedded in each other.
    if a_max_w > b_min_w + flush_tol and a_min_w < b_max_w - flush_tol:
        #print('Deep Volume')
        return [-2, -2]

    # 2. Flush Contact: A is physically "behind" B (within mesh noise)
    if abs(a_max_w - b_min_w) <= flush_tol:
        # A pushing forward (+W) hits B ONLY if A faces +W and B faces -W
        if normal_a_w > ANGLE_NORMAL_TOL and normal_b_w < -ANGLE_NORMAL_TOL:
            #print('Flush and crash (+)')
            result[0] = 2

    # 3. Flush Contact: A is physically "ahead" of B (within mesh noise)
    if abs(a_min_w - b_max_w) <= flush_tol:
        # A pushing backward (-W) hits B ONLY if A faces -W and B faces +W
        if normal_a_w < -ANGLE_NORMAL_TOL and normal_b_w > ANGLE_NORMAL_TOL:
            #print('Flush and crash (-)')
            result[1] = 2

    # 4. Trailing Edge Checks (A is entirely behind/ahead with a clear gap > 0.05)
    if b_min_w > a_max_w + flush_tol:
        # A is behind B. A moves +W to cross the gap and hit B.
        if normal_a_w > ANGLE_NORMAL_TOL and normal_b_w < -ANGLE_NORMAL_TOL:
            #print('Crash (+)')
            result[0] = 2 
            
    if a_min_w > b_max_w + flush_tol:
        # A is ahead of B. A moves -W to cross the gap and hit B.
        if normal_a_w < -ANGLE_NORMAL_TOL and normal_b_w > ANGLE_NORMAL_TOL:
            #print('Crash (-)')
            result[1] = 2  

    return result
    
def check_PF_overlap_cython(pf_a: PseudoFace, pf_b: PseudoFace, flush_tol = FLUSH_TOL, angle_tol = ANGLE_NORMAL_TOL):
    result = [0, 0] 
    w_idx = pf_a.extraction_axis

    a_min_u, a_min_v = pf_a.triangles_2d.min(axis=(0,1))
    a_max_u, a_max_v = pf_a.triangles_2d.max(axis=(0,1))
    a_min_w = pf_a.triangles_3d[:, :, w_idx].min()
    a_max_w = pf_a.triangles_3d[:, :, w_idx].max()

    b_min_u, b_min_v = pf_b.triangles_2d.min(axis=(0,1))
    b_max_u, b_max_v = pf_b.triangles_2d.max(axis=(0,1))
    b_min_w = pf_b.triangles_3d[:, :, w_idx].min()
    b_max_w = pf_b.triangles_3d[:, :, w_idx].max()

    overlap_min_u = max(a_min_u, b_min_u)
    overlap_max_u = min(a_max_u, b_max_u)
    overlap_min_v = max(a_min_v, b_min_v)
    overlap_max_v = min(a_max_v, b_max_v)

    if not ((overlap_min_u <= overlap_max_u) and (overlap_min_v <= overlap_max_v)):
        return [0, 0] 
    
    a_lims = [(a_min_u, a_max_u), (a_min_v, a_max_v)]
    b_lims = [(b_min_u, b_max_u), (b_min_v, b_max_v)]
    coaabb_overlap = check_COAABB_overlap(a_lims, b_lims)

    if not coaabb_overlap:
        return [0, 0] 

    # ---> EXACT CYTHON OVERLAP FILTER (EARLY EXIT) <---
    # We cast to float64 contiguous arrays only when absolutely necessary
    tris_a = np.ascontiguousarray(pf_a.triangles_2d, dtype=np.float64)
    tris_b = np.ascontiguousarray(pf_b.triangles_2d, dtype=np.float64)
    
    # Let C instantly mathematically prove if ANY triangles touch
    if not fast_any_intersection_c(tris_a, tris_b):
        return [0, 0]

    # -------------------------------------------------------------
    # THE KINEMATIC DIRECTIONAL CHECK
    # -------------------------------------------------------------
    normal_a_w = pf_a.part.face_normals[pf_a.face_indices[0]][w_idx]
    normal_b_w = pf_b.part.face_normals[pf_b.face_indices[0]][w_idx]

    # FLUSH TOLERANCE: 
    # DISTANCE_TOL (0.0002) is too strict for tessellated CAD. 
    # We use 0.05mm to absorb the natural mesh "criss-crossing" of flush parts.
    #flush_tol = FLUSH_TOL

    # 1. Deep Volume Overlap 
    # If they overlap by MORE than the flush tolerance, they are truly embedded in each other.
    if a_max_w > b_min_w + flush_tol and a_min_w < b_max_w - flush_tol:
        #print('Deep Volume')
        return [-2, -2]

    # 2. Flush Contact: A is physically "behind" B (within mesh noise)
    if abs(a_max_w - b_min_w) <= flush_tol:
        # A pushing forward (+W) hits B ONLY if A faces +W and B faces -W
        if normal_a_w > ANGLE_NORMAL_TOL and normal_b_w < -ANGLE_NORMAL_TOL:
            #print('Flush and crash (+)')
            result[0] = 2

    # 3. Flush Contact: A is physically "ahead" of B (within mesh noise)
    if abs(a_min_w - b_max_w) <= flush_tol:
        # A pushing backward (-W) hits B ONLY if A faces -W and B faces +W
        if normal_a_w < -ANGLE_NORMAL_TOL and normal_b_w > ANGLE_NORMAL_TOL:
            #print('Flush and crash (-)')
            result[1] = 2

    # 4. Trailing Edge Checks (A is entirely behind/ahead with a clear gap > 0.05)
    if b_min_w > a_max_w + flush_tol:
        # A is behind B. A moves +W to cross the gap and hit B.
        if normal_a_w > ANGLE_NORMAL_TOL and normal_b_w < -ANGLE_NORMAL_TOL:
            #print('Crash (+)')
            result[0] = 2 
            
    if a_min_w > b_max_w + flush_tol:
        # A is ahead of B. A moves -W to cross the gap and hit B.
        if normal_a_w < -ANGLE_NORMAL_TOL and normal_b_w > ANGLE_NORMAL_TOL:
            #print('Crash (-)')
            result[1] = 2  

    return result

def focus_facet_intersection_test(pf_a: PseudoFace, pf_b: PseudoFace, direction: str):
    """Checks if any of the focus facets of PseudoFace of part A intersects with any of those of part B.
    Direction is either '+w' or '-w' depending on whether we are checking the positive or negative extraction direction
    Returns:
        0 if no collision detected between any of the focus facets.
        1 if A cannot be extracted in the positive direction without colliding with B, but can be extracted in the negative direction.
        -1 if A cannot be extracted in the negative direction without colliding with B, but can be extracted in the positive direction"""
    
    # First we test the AABBs of the candidates in 2D
    for facet_a in pf_a.focus_facets:
        coords_2d_a = pf_a.triangles_2d[facet_a]
        min_u_a, min_v_a = coords_2d_a.min(axis=0)
        max_u_a, max_v_a = coords_2d_a.max(axis=0)

        for facet_b in pf_b.focus_facets:
            coords_2d_b = pf_b.triangles_2d[facet_b]
            min_u_b, min_v_b = coords_2d_b.min(axis=0)
            max_u_b, max_v_b = coords_2d_b.max(axis=0)

            # 1. Check if the AABBs of the facets overlap in 2D
            # If the boxes don't overlap in U or V, they can't touch!
            if (min_u_a > max_u_b or max_u_a < min_u_b or
                min_v_a > max_v_b or max_v_a < min_v_b):
                continue # Skip to the next pair instantly

            # 2. Check 2D polygon intersection
            poly_a = Polygon(coords_2d_a)
            poly_b = Polygon(coords_2d_b)

            if not poly_a.intersects(poly_b):
                continue # If the 2D projections don't intersect, skip to the next pair

            # 3. If theres a 2D intersection we check the depth to see if they collide in +/- w
            extraction_axis = pf_a.extraction_axis
            min_w_a = pf_a.triangles_3d[facet_a][:, extraction_axis].min()
            max_w_a = pf_a.triangles_3d[facet_a][:, extraction_axis].max()
            min_w_b = pf_b.triangles_3d[facet_b][:, extraction_axis].min()
            max_w_b = pf_b.triangles_3d[facet_b][:, extraction_axis].max()

            # Case a: Static overlap
            if max_w_a >= min_w_b and min_w_a <= max_w_b:
                return 2 # Collide instantly in both directions
            
            # Case b: I need to extract A in the positive direction, so I check if B is blocking that
            elif max_w_a <= min_w_b and direction == "+w":
                return 1 # A cannot be extracted in the positive direction without colliding with B
        
            # Case c: I need to extract A in the negative direction, so I check if B is blocking that
            elif min_w_a >= max_w_b and direction == "-w":
                return -1 # A cannot be extracted in the negative direction without colliding with B
            
    return 0 # No collision detected between any of the focus facets

def focus_facet_intersection_full(pseudo_faces_a, pseudo_faces_b):
    extraction_axis = pseudo_faces_a[0].extraction_axis
    pos_result = 0
    neg_result = 0
    for pf_a in pseudo_faces_a:
        for pf_b in pseudo_faces_b:
                ff_intersection_pos = focus_facet_intersection_test(pf_a, pf_b, "+w")
                ff_intersection_neg = focus_facet_intersection_test(pf_a, pf_b, "-w")

                if ff_intersection_pos == 2 or ff_intersection_neg == 2:
                    return 1, 1
                else:
                    if ff_intersection_pos == 1:
                        pos_result = 1
                    elif ff_intersection_neg == 1:
                        neg_result = 1
    return pos_result, neg_result


## Determine if parts intersect their AABBs
def check_3D_AABB_intersection(part_a, part_b):
    a_min_3d = part_a.bounds[0]
    a_max_3d = part_a.bounds[1]
    b_min_3d = part_b.bounds[0]
    b_max_3d = part_b.bounds[1]

    # Evaluate intersection boolean
    intersects = not (np.any(a_min_3d > b_max_3d) or np.any(a_max_3d < b_min_3d))
    
    # Always return the tuples, just flip the boolean flag!
    return [(a_min_3d, a_max_3d), (b_min_3d, b_max_3d), intersects]


## Narrow Phase Test functions (facet intersection)
def check_static_interference(part_a, part_b):
    "Checks if part_a and part_b are already colliding in their current position, means that they statically interfere"
    collision_manager = trimesh.collision.CollisionManager()
    collision_manager.add_object('part_a', part_a)
    collision_manager.add_object('part_b', part_b)

    is_colliding = collision_manager.in_collision_internal()
    return is_colliding


def filter_facets(pf_a: PseudoFace, pf_b: PseudoFace, AABB_3d_intersection, only_focus_facets = False, tolerance = 1e-4):
    # Pass all valid PseudoFace triangles directly to the Narrow Phase 
    # without the broken Z-bounds logic deleting them!
    
    candidates_a = list(pf_a.focus_facets) if only_focus_facets else list(range(len(pf_a.triangles_3d)))
    candidates_b = list(pf_b.focus_facets) if only_focus_facets else list(range(len(pf_b.triangles_3d)))

    return candidates_a, candidates_b



def hybrid_facet_intersection_test(poly_a: Polygon, poly_b: Polygon, use_MRT, MRT_tolerance = 1e-4):
    # We use the dynamic 2D polygons passed from the Narrow Phase.
    # This permanently destroys the hardcoded [:, :2] XY bug!
    if not poly_a.intersects(poly_b):
        return 0 

    if use_MRT:
        overlap_poly = poly_a.intersection(poly_b)
        
        # Failsafe for 0-area overlapping lines
        if overlap_poly.is_empty or overlap_poly.area < 1e-8:
            return 1 
        
        min_u, min_v, max_u, max_v = overlap_poly.bounds
        overlap_distance = min(max_u - min_u, max_v - min_v)
            
        if overlap_distance < MRT_tolerance:
            return 1 
        
        return 2
    
    return 2

# OK
def get_primitive_points(poly_a: Polygon, poly_b: Polygon):
    if not poly_a.intersects(poly_b):
        return np.empty((0, 2))

    overlap = poly_a.intersection(poly_b)
    raw_coords = []

    # Shapely can sometimes return complex geometries when buffering triangles
    if isinstance(overlap, Polygon):
        raw_coords.extend(list(overlap.exterior.coords)[:-1])
    elif isinstance(overlap, MultiPolygon):
        for poly in overlap.geoms:
            raw_coords.extend(list(poly.exterior.coords)[:-1])
    else:
        # Catches LineStrings, Points, or GeometryCollections that survive the buffer
        if hasattr(overlap, 'coords'):
            raw_coords.extend(list(overlap.coords))
        elif hasattr(overlap, 'geoms'):
            for geom in overlap.geoms:
                if hasattr(geom, 'coords'):
                    raw_coords.extend(list(geom.coords))
                elif hasattr(geom, 'exterior'): # Catch Polygons inside Collections
                    raw_coords.extend(list(geom.exterior.coords)[:-1])

    if len(raw_coords) > 0:
        unique_pts = np.unique(np.array(raw_coords), axis=0)
        return unique_pts

    return np.empty((0, 2))

def primitive_point_projection(pf, facet_idx, primitive_points):
    global_idx = pf.face_indices[facet_idx]

    # THE FIX: It's already an integer!
    w_idx = pf.extraction_axis
    axes = [0, 1, 2]
    axes.remove(w_idx)
    u_idx, v_idx = axes[0], axes[1]

    nu = pf.part.face_normals[global_idx][u_idx]
    nv = pf.part.face_normals[global_idx][v_idx]
    nw = pf.part.face_normals[global_idx][w_idx]

    u0 = pf.triangles_3d[facet_idx][0, u_idx]
    v0 = pf.triangles_3d[facet_idx][0, v_idx]
    w0 = pf.triangles_3d[facet_idx][0, w_idx]

    D = -(nu * u0 + nv * v0 + nw * w0)

    if abs(nw) < 1e-6:
        projected_w = np.full(primitive_points.shape[0], w0)
    else:
        projected_w = -(nu * primitive_points[:, 0] + nv * primitive_points[:, 1] + D) / nw

    projected_points_3d = np.zeros((primitive_points.shape[0], 3))
    projected_points_3d[:, u_idx] = primitive_points[:, 0]
    projected_points_3d[:, v_idx] = primitive_points[:, 1]
    projected_points_3d[:, w_idx] = projected_w

    return projected_points_3d

def IM_entry_calculation(pf_a, facet_idx_a, pf_b, facet_idx_b, primitive_points_a, primitive_points_b, interference_type, w_tol = W_TOL, n_tol = ANGLE_NORMAL_TOL, use_MRT = False):
    global_idx_a = pf_a.face_indices[facet_idx_a]
    global_idx_b = pf_b.face_indices[facet_idx_b]

    # THE FIX: It's already an integer!
    w_idx = pf_a.extraction_axis 

    normal_a = pf_a.part.face_normals[global_idx_a][w_idx]
    normal_b = pf_b.part.face_normals[global_idx_b][w_idx]
    #print(f'Normal A: {normal_a}, Normal B: {normal_b}')
    a_ij, a_ji = 0, 0
    
    #w_tol = 0.0002 # 0.05mm depth tolerance prevents CAD micro-overlap errors
    # w_tol = W_TOL
    # n_tol = ANGLE_NORMAL_TOL # 0.05 normal tolerance (~3 degrees) mathematically ignores mesh noise on sliding rails

    for i in range(primitive_points_a.shape[0]):
        w_prim_a = primitive_points_a[i][w_idx]
        w_prim_b = primitive_points_b[i][w_idx]

        # A points +W, B points -W (e.g. Lid pushes +X into Base -X stopper)
        if normal_a > n_tol and normal_b < -n_tol:
            # A hits B if A is physically behind B or flush with B
            if w_prim_a <= w_prim_b + w_tol:
                a_ij = interference_type

        # A points -W, B points +W (e.g. Lid pushes -X into Base +X stopper)
        elif normal_a < -n_tol and normal_b > n_tol:
            # A hits B if A is physically in front of B or flush with B
            if w_prim_a >= w_prim_b - w_tol:
                a_ji = interference_type

        # Once both directions are blocked for this pair, no need to keep checking points
        if a_ij != 0 and a_ji != 0: break

    return a_ij, a_ji

def narrow_phase_chunk_worker(chunk, pf_a, pf_b, use_MRT, w_tol, n_tol):
    max_pos, max_neg = 0, 0
    
    for idx_a, idx_b in chunk:
        min_a = pf_a.triangles_2d[idx_a].min(axis=0)
        max_a = pf_a.triangles_2d[idx_a].max(axis=0)
        min_b = pf_b.triangles_2d[idx_b].min(axis=0)
        max_b = pf_b.triangles_2d[idx_b].max(axis=0)

        if (min_a[0] > max_b[0] + 1e-5 or max_a[0] < min_b[0] - 1e-5 or
            min_a[1] > max_b[1] + 1e-5 or max_a[1] < min_b[1] - 1e-5):
            continue

        poly_a = Polygon(pf_a.triangles_2d[idx_a])
        poly_b = Polygon(pf_b.triangles_2d[idx_b])
        
        buf_tol = 1e-4
        poly_a_buf = poly_a.buffer(buf_tol)
        poly_b_buf = poly_b.buffer(buf_tol)

        if not poly_a_buf.intersects(poly_b_buf):
            continue
            
        hybrid_result = hybrid_facet_intersection_test(poly_a_buf, poly_b_buf, use_MRT)
        
        if hybrid_result not in [1, 2]:
            continue

        primitive_all = get_primitive_points(poly_a_buf, poly_b_buf)
        if len(primitive_all) == 0:
            continue

        primitive_points_a = primitive_point_projection(pf_a, idx_a, primitive_all)
        primitive_points_b = primitive_point_projection(pf_b, idx_b, primitive_all)
        
        positive_entry, negative_entry = IM_entry_calculation(
            pf_a, idx_a, pf_b, idx_b, primitive_points_a, primitive_points_b, hybrid_result, w_tol, n_tol, use_MRT
        )

        max_pos = max(max_pos, positive_entry)
        max_neg = max(max_neg, negative_entry)

        if max_pos == 2 and max_neg == 2:
            break 

    return max_pos, max_neg

def evaluate_narrow_phase_parallel(candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT, w_tol = W_TOL, n_tol = ANGLE_NORMAL_TOL):
    
    all_pairs = list(product(candidates_a, candidates_b))
    total_pairs = len(all_pairs)
    
    if total_pairs == 0:
        return 0, 0
    
    # Gatekeeper: Serial for small workloads
    if total_pairs < 2000:
        return narrow_phase_chunk_worker(all_pairs, pf_a, pf_b, use_MRT, w_tol, n_tol)
        
    # Chunking Logic for massive workloads
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, total_pairs // num_cores)
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
    
    worker_args = [[chunk, pf_a, pf_b, use_MRT, w_tol, n_tol] for chunk in chunks]
    global_max_pos, global_max_neg = 0, 0
    
    for res_pos, res_neg in fast_parallel_execute(narrow_phase_chunk_worker, worker_args, num_proc=num_cores):
        global_max_pos = max(global_max_pos, res_pos)
        global_max_neg = max(global_max_neg, res_neg)
        if global_max_pos == 2 and global_max_neg == 2:
            break
                
    return global_max_pos, global_max_neg

def evaluate_narrow_phase_parallel(candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT, w_tol = W_TOL, n_tol = ANGLE_NORMAL_TOL):
    
    all_pairs = list(product(candidates_a, candidates_b))
    total_pairs = len(all_pairs)
    
    if total_pairs == 0:
        return 0, 0
    
    # Gatekeeper: Serial for small workloads
    if total_pairs < 2000:
        return narrow_phase_chunk_worker(all_pairs, pf_a, pf_b, use_MRT, w_tol, n_tol)
        
    # Chunking Logic for massive workloads
    num_cores = multiprocessing.cpu_count()
    chunk_size = max(1, total_pairs // num_cores)
    chunks = [all_pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]
    
    worker_args = [[chunk, pf_a, pf_b, use_MRT, w_tol, n_tol] for chunk in chunks]
    global_max_pos, global_max_neg = 0, 0
    
    for res_pos, res_neg in fast_parallel_execute(narrow_phase_chunk_worker, worker_args, num_proc=num_cores):
        global_max_pos = max(global_max_pos, res_pos)
        global_max_neg = max(global_max_neg, res_neg)
        if global_max_pos == 2 and global_max_neg == 2:
            break
                
    return global_max_pos, global_max_neg

def evaluate_narrow_phase(candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT, w_tol = W_TOL, n_tol = ANGLE_NORMAL_TOL,
                          abort_threshold = None):
    max_pos, max_neg = 0, 0
    heavy_checks = 0 # <--- NEW COUNTER

    for idx_a, idx_b in product(candidates_a, candidates_b):
        min_a = pf_a.triangles_2d[idx_a].min(axis=0)
        max_a = pf_a.triangles_2d[idx_a].max(axis=0)
        min_b = pf_b.triangles_2d[idx_b].min(axis=0)
        max_b = pf_b.triangles_2d[idx_b].max(axis=0)

        if (min_a[0] > max_b[0] + 1e-5 or max_a[0] < min_b[0] - 1e-5 or
            min_a[1] > max_b[1] + 1e-5 or max_a[1] < min_b[1] - 1e-5):
            continue

        # ---> 2. THE DEEP ABORT TRIGGER <---
        # Only counts pairs that actually force heavy Shapely operations
        if abort_threshold is not None:
            heavy_checks += 1
            if heavy_checks > abort_threshold:
                return -999, -999

        poly_a = Polygon(pf_a.triangles_2d[idx_a])
        poly_b = Polygon(pf_b.triangles_2d[idx_b])
        
        # Micro-buffer to force tangents to overlap
        buf_tol = 1e-4
        poly_a_buf = poly_a.buffer(buf_tol)
        poly_b_buf = poly_b.buffer(buf_tol)

        if not poly_a_buf.intersects(poly_b_buf):
            continue
            
        overlap_poly = poly_a_buf.intersection(poly_b_buf)
        
        # Remove the strict area < 1e-5 check. 
        # The buffer guarantees that if they touch, there is geometry.
        if overlap_poly.is_empty:
            continue

        # ---> THE CONSISTENCY FIX <---
        # ---> FIX: MUST USE THE BUFFERED POLYGONS SO THEY ACTUALLY TOUCH! <---
        hybrid_result = hybrid_facet_intersection_test(poly_a_buf, poly_b_buf, use_MRT)
        
        if hybrid_result not in [1, 2]:
            continue

        # ---> FIX: GET POINTS FROM THE BUFFERED OVERLAP <---
        primitive_all = get_primitive_points(poly_a_buf, poly_b_buf)
        if len(primitive_all) == 0:
            continue

        primitive_points_a = primitive_point_projection(pf_a, idx_a, primitive_all)
        primitive_points_b = primitive_point_projection(pf_b, idx_b, primitive_all)
        
        positive_entry, negative_entry = IM_entry_calculation(
            pf_a, idx_a, pf_b, idx_b, primitive_points_a, primitive_points_b, hybrid_result, w_tol, n_tol, use_MRT
        )

        max_pos = max(max_pos, positive_entry)
        max_neg = max(max_neg, negative_entry)

        if max_pos == 2 and max_neg == 2:
            break 

    return max_pos, max_neg

def evaluate_narrow_phase_cython(candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT, w_tol=W_TOL, n_tol=ANGLE_NORMAL_TOL, abort_threshold=None, mrt_tol=1e-4):
    
    tris_a_2d = np.ascontiguousarray(pf_a.triangles_2d[candidates_a], dtype=np.float64)
    tris_b_2d = np.ascontiguousarray(pf_b.triangles_2d[candidates_b], dtype=np.float64)
    
    intersecting_pairs = get_intersecting_pairs_c(tris_a_2d, tris_b_2d)
    
    if not intersecting_pairs:
        return 0, 0
        
    if abort_threshold is not None and len(intersecting_pairs) > abort_threshold:
        return -999, -999
        
    tris_a_3d = np.ascontiguousarray(pf_a.triangles_3d[candidates_a], dtype=np.float64)
    tris_b_3d = np.ascontiguousarray(pf_b.triangles_3d[candidates_b], dtype=np.float64)
    
    global_indices_a = pf_a.face_indices[candidates_a]
    global_indices_b = pf_b.face_indices[candidates_b]
    
    normals_a = np.ascontiguousarray(pf_a.part.face_normals[global_indices_a], dtype=np.float64)
    normals_b = np.ascontiguousarray(pf_b.part.face_normals[global_indices_b], dtype=np.float64)
    
    w_idx = pf_a.extraction_axis
    axes = [0, 1, 2]
    axes.remove(w_idx)
    u_idx, v_idx = axes[0], axes[1]
    
    max_pos, max_neg = evaluate_deep_narrow_phase_c(
        tris_a_2d, tris_b_2d,
        tris_a_3d, tris_b_3d,
        normals_a, normals_b,
        intersecting_pairs,
        w_idx, u_idx, v_idx,
        w_tol, n_tol,
        use_MRT, mrt_tol       # <--- Pass the new variables to C!
    )
    
    return max_pos, max_neg

## Main Extraction functions
def evaluate_pair_interference(part_a_data, part_b_data, extraction_axis,
                               override_dist_tol=None, override_w_tol=None, override_flush_tol=None, override_n_tol=None,
                               abort_threshold=None, use_parallel_narrow=False, use_cython=True):
    """Evaluates the maximum interference between two parts along a specific axis."""

    # Fallback to your strict defaults if no override is passed
    dist_tol = override_dist_tol if override_dist_tol is not None else DISTANCE_TOL
    w_tol = override_w_tol if override_w_tol is not None else W_TOL
    flush_tol = override_flush_tol if override_flush_tol is not None else FLUSH_TOL
    n_tol = override_n_tol if override_n_tol is not None else ANGLE_NORMAL_TOL

    to_origin_A = part_a_data["to_origin"]
    
    # ---> 1. FAST NATIVE NUMPY TRANSFORMATIONS (NO TRIMESH OVERHEAD) <---
    tris_a, norms_a = transform_mesh_data(part_a_data["triangles"], part_a_data["face_normals"], to_origin_A)
    part_a_fast = FastPart(tris_a, norms_a)
    
    tris_b, norms_b = transform_mesh_data(part_b_data["triangles"], part_b_data["face_normals"], to_origin_A)
    part_b_fast = FastPart(tris_b, norms_b)

    overlap_region, overlap_result = check_2d_aabb_overlap(
        part_a_fast.bounds, part_b_fast.bounds, extraction_axis, dist_tol)
    
    if overlap_result == 0: return 0, 0
    if overlap_result == -1: return 0, 2
    if overlap_result == 1: return 2, 0
    if overlap_result == 2: return 2, 2

    parts_AABB_interfere = check_3D_AABB_intersection(part_a_fast, part_b_fast)
    
    part_a_aux, part_b_aux = None, None  # <--- INITIALIZE THEM HERE
    
    if use_cython:
        use_MRT = parts_AABB_interfere[2] 
    else:
        # ---> ONLY COPY TRIMESH IF THE SLOW PYTHON FALLBACK IS NEEDED <---
        part_a_aux = part_a_data["part_mesh"].copy()
        part_b_aux = part_b_data["part_mesh"].copy()
        part_a_aux.apply_transform(to_origin_A)
        part_b_aux.apply_transform(to_origin_A)
        use_MRT = check_static_interference(part_a_aux, part_b_aux)
        if parts_AABB_interfere[2] == True:
            use_MRT = True

    # 2. PseudoFace Generation (Now passing our FastPart!)
    pseudo_faces_a = create_PFs(part_a_fast, extraction_axis, part_a_data["face_adjacency"])
    pseudo_faces_b = create_PFs(part_b_fast, extraction_axis, part_b_data["face_adjacency"])
    
    for pf_a in pseudo_faces_a: pf_a.get_focus_facets(overlap_region)
    for pf_b in pseudo_faces_b: pf_b.get_focus_facets(overlap_region)

    max_pos, max_neg = 0, 0
    full_interference = False
    

    current_pf_intersect = [0, 0]
    
    for pf_a, pf_b in product(pseudo_faces_a, pseudo_faces_b):
        if full_interference:
            #print(f'\tFull PF Interference Detected')
            break
            
        # ---> MASTER SWITCH 1: THE GATEKEEPER <---
        if use_cython:
            pf_intersect = check_PF_overlap_cython(pf_a, pf_b, flush_tol)
        else:
            pf_intersect = check_PF_overlap_original(pf_a, pf_b, flush_tol)
        
        final_pos, final_neg = pf_intersect
        if final_pos == 2 and final_pos != current_pf_intersect[0] and -2 not in pf_intersect:
            current_pf_intersect[0] = final_pos
            #print(f'\tPF Overlap Update: max_pos {final_pos}')
        if final_neg == 2 and final_neg != current_pf_intersect[1] and -2 not in pf_intersect:
            current_pf_intersect[1] = final_neg
            #print(f'\tPF Overlap Update: max_neg {final_neg}')

        if -2 in pf_intersect:
            for attempt in ["focus_facets", "full_fallback"]:
                is_focus = (attempt == "focus_facets")
                candidates_a, candidates_b = filter_facets(
                    pf_a, pf_b, parts_AABB_interfere, only_focus_facets=is_focus
                )
                
                if not candidates_a or not candidates_b: continue

                # ---> THE INSTANT ABORT <---
                # Costs 0 seconds. Saves 45 seconds on the Stool.
                if attempt == "full_fallback" and abort_threshold is not None:
                    if (len(candidates_a) * len(candidates_b)) > abort_threshold:
                        return -999, -999

                # ---> MASTER SWITCH 2: THE DEEP NARROW PHASE <---
                if use_cython:
                    c_pos, c_neg = evaluate_narrow_phase_cython(
                        candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT, w_tol, n_tol,
                        abort_threshold = abort_threshold
                    )
                elif use_parallel_narrow:
                    c_pos, c_neg = evaluate_narrow_phase_parallel(
                        candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT, w_tol, n_tol
                    )
                else:
                    c_pos, c_neg = evaluate_narrow_phase(
                        candidates_a, candidates_b, pf_a, pf_b, part_a_aux, part_b_aux, use_MRT, w_tol, n_tol,
                        abort_threshold = abort_threshold
                    )

                if c_pos == -999 or c_neg == -999:
                    return -999, -999
                
                final_pos, final_neg = max(final_pos, c_pos), max(final_neg, c_neg)
                if final_pos == 2 and final_neg == 2: 
                    break 
        
        max_pos, max_neg = max(max_pos, final_pos), max(max_neg, final_neg)
        if max_pos == 2 and max_neg == 2:
            full_interference = True
            break
    
    return max_pos, max_neg

def calculate_IM_matrices(assembly_manifest, use_parallel_narrow=False, use_cython=True):
    N = len(assembly_manifest)
    matrices = {d: np.zeros((N, N), dtype=int) for d in ["+x", "-x", "+y", "-y", "+z", "-z"]}
    axis_matrix_map = {"x": ["+x", "-x"], "y": ["+y", "-y"], "z": ["+z", "-z"]}
    part_keys = list(assembly_manifest.keys())
    print(f'Calculating Interference Matrices...')
    for extraction_axis, (pos_key, neg_key) in axis_matrix_map.items():
        #print(f'\n----------------- Checking {extraction_axis} Direction -----------------')
        
        for i, j in permutations(range(N), 2):
            part_a_name = part_keys[i]
            part_b_name = part_keys[j]
            
            part_a_data = assembly_manifest[part_a_name]
            part_b_data = assembly_manifest[part_b_name]

            # if 'CameraBase_v2-3-1' not in {part_a_name, part_b_name} :
            #     continue

            # Log the pair checking
            #print(f'Moving: {part_a_name}, Static: {part_b_name}')

            # Names perfectly synced with the updated helper function
            pos_val, neg_val = evaluate_pair_interference(
                part_a_data, part_b_data, extraction_axis, 
                use_parallel_narrow=use_parallel_narrow, 
                use_cython=use_cython
            )
            
            matrices[pos_key][i, j] = pos_val
            matrices[neg_key][i, j] = neg_val
    print(f'Done calculating Interference Matrices.')
    return matrices

## NEW: Optimized Action Row Generation
def evaluate_optimized_action(part_a_mesh, part_b_mesh, raw_opt_action, tol=1e-5):
    """
    Evaluates interference along an arbitrary optimized vector by sanitizing the vector,
    aligning it to the local +X axis, and calling the standard 2D check.
    """
    # 1. Sanitize the physics engine noise
    opt_action = np.array(raw_opt_action, dtype=float)
    opt_action[np.abs(opt_action) < tol] = 0.0
    opt_action[np.abs(opt_action - 1.0) < tol] = 1.0
    opt_action[np.abs(opt_action + 1.0) < tol] = -1.0
    
    # Failsafe: Ensure it didn't collapse to zero
    if np.linalg.norm(opt_action) < 1e-6:
        raise ValueError("opt_action collapsed to zero after snapping.")

    # 2. Normalize strictly to 1.0 to become our new +X axis
    v_x = opt_action / np.linalg.norm(opt_action)
    
    # 3. Build orthogonal Y and Z axes safely
    ref_vec = np.array([0.0, 1.0, 0.0])
    
    # If the extraction is perfectly parallel to Y, switch the reference to Z
    if np.abs(np.dot(v_x, ref_vec)) > 0.99:
        ref_vec = np.array([0.0, 0.0, 1.0])
        
    v_z = np.cross(v_x, ref_vec)
    v_z /= np.linalg.norm(v_z)
    
    v_y = np.cross(v_z, v_x)
    # Note: v_y is naturally normalized because v_x and v_z are orthogonal unit vectors
    
    # 4. Create the 'to_origin' rotation matrix
    to_origin = np.eye(4)
    to_origin[0, :3] = v_x
    to_origin[1, :3] = v_y
    to_origin[2, :3] = v_z
    
    # 5. Add translation to center Part A at the origin
    center = part_a_mesh.bounding_box.centroid
    to_origin[:3, 3] = -np.dot(to_origin[:3, :3], center)
    
    # 6. Mock the data dictionaries exactly as your pipeline expects
    temp_a_data = {
        "part_mesh": part_a_mesh,
        "to_origin": to_origin,
        "face_adjacency": part_a_mesh.face_adjacency,
        "triangles": part_a_mesh.triangles.copy(),      # <--- ADD THIS
        "face_normals": part_a_mesh.face_normals.copy() # <--- ADD THIS
    }
    
    temp_b_data = {
        "part_mesh": part_b_mesh,
        "to_origin": np.eye(4),
        "face_adjacency": part_b_mesh.face_adjacency,
        "triangles": part_b_mesh.triangles.copy(),      # <--- ADD THIS
        "face_normals": part_b_mesh.face_normals.copy() # <--- ADD THIS
    }
    
    # 7. Run the existing evaluation strictly along the localized "x" axis
    return evaluate_pair_interference(temp_a_data, temp_b_data, "x")

def get_optimized_action_row(part_name, assembly_manifest, optimized_vector):
    part_keys = list(assembly_manifest.keys())

    part_a_data = assembly_manifest[part_name]
    part_a_idx = part_a_data["matrix_idx"]
    part_a_mesh = part_a_data["part_mesh"]

    matrix_row = np.zeros(len(part_keys))

    for j, part_b_name in enumerate(part_keys):
        if part_a_idx == j:
            matrix_row[j] = 0
            continue

        part_b_data = assembly_manifest[part_b_name]
        part_b_mesh = part_b_data["part_mesh"]
        entry, _ = evaluate_optimized_action(part_a_mesh, part_b_mesh, optimized_vector)

        matrix_row[j] = entry

    return matrix_row

## NEW: Check collision with the ground plane
def identify_ground_parts(assembly_manifest, z_axis_index=2, tolerance=1e-3):
    """
    Instantly identifies which parts are touching the ground by finding 
    the lowest Z-coordinate in the global bounding boxes.
    """
    # 1. Extract the minimum Z value for every part
    # bounds[0] is the minimums [min_x, min_y, min_z]
    part_z_mins = {}
    for part_name, data in assembly_manifest.items():
        z_min = data['part_mesh'].bounds[0][z_axis_index]
        part_z_mins[part_name] = z_min
        
    # 2. Find the absolute lowest point in the entire assembly
    global_ground_z = min(part_z_mins.values())
    
    # 3. Any part within the tolerance of the ground floor is a base part
    parts_on_ground = [
        name for name, z_min in part_z_mins.items() 
        if abs(z_min - global_ground_z) <= tolerance
    ]
    
    return parts_on_ground

## NEW: Get freedom score from each part based on interference matrices
def get_free_directions(part_id, current_assembly_ids, part_ids_list, matrices_dict):
    part_idx = part_ids_list.index(part_id)
    fixed_indices = [
        part_ids_list.index(pid) for pid in current_assembly_ids if pid != part_id
    ]
    
    free_dirs = []
    if not fixed_indices:
        return ["+z"] # Default to extracting upwards if it's the last part
        
    for direction in ["+x", "-x", "+y", "-y", "+z", "-z"]:
        matrix = matrices_dict[direction]
        if np.sum(matrix[part_idx, fixed_indices]) == 0:
            free_dirs.append(direction)
            
    return free_dirs


def get_direction_string(action_vec):
    """Maps a 3D numpy vector to the string keys used in the interference matrices."""
    if np.allclose(action_vec, [1, 0, 0]): return '+x'
    if np.allclose(action_vec, [-1, 0, 0]): return '-x'
    if np.allclose(action_vec, [0, 1, 0]): return '+y'
    if np.allclose(action_vec, [0, -1, 0]): return '-y'
    if np.allclose(action_vec, [0, 0, 1]): return '+z'
    if np.allclose(action_vec, [0, 0, -1]): return '-z'
    return None

def get_freedom_score(part_id, current_assembly_ids, part_ids_list, matrices_dict):
    """
    Calculates how many global axes a part can be extracted along without hitting
    the remaining parts in the assembly.
    """
    # 1. Get the matrix index of the part we want to evaluate
    part_idx = part_ids_list.index(part_id)
    
    # 2. Get the matrix indices of all OTHER parts still in the assembly
    fixed_indices = [
        part_ids_list.index(pid) for pid in current_assembly_ids if pid != part_id
    ]
    
    # If it's the last part, it's completely free
    if not fixed_indices:
        return 6
        
    score = 0
    # 3. Check all 6 directions using your dictionary keys
    for direction in ["+x", "-x", "+y", "-y", "+z", "-z"]:
        matrix = matrices_dict[direction]
        
        # matrix[part_idx, fixed_indices] slices the row for the moving part,
        # looking ONLY at the columns for the parts that are still assembled.
        # If the sum is 0, there are no collisions in that direction!
        if np.sum(matrix[part_idx, fixed_indices]) == 0:
            score += 1
            
    return score

## Data Handling
def clean_obb_matrix(to_origin, tolerance=0.05):
    """
    Snaps microscopic noise to 0/1, then uses Singular Value Decomposition (SVD) 
    to guarantee the resulting matrix is a perfectly orthogonal 3D rotation matrix.
    """
    matrix = to_origin.copy()
    rot = matrix[:3, :3]
    
    # 1. Snap the microscopic noise on intended flush axes
    rot[np.abs(rot) < tolerance] = 0.0
    rot[np.abs(rot - 1.0) < tolerance] = 1.0
    rot[np.abs(rot + 1.0) < tolerance] = -1.0
    
    # 2. SVD Re-Orthogonalization 
    # This takes the snapped matrix and mathematically forces the axes to be exactly 
    # 90 degrees apart and length 1.0, completely preventing CAD mesh warping!
    U, _, Vt = np.linalg.svd(rot)
    perfect_rot = np.dot(U, Vt)
    
    # 3. Failsafe: Ensure it's a true rotation (determinant of +1) and not a reflection
    if np.linalg.det(perfect_rot) < 0:
        Vt[2, :] *= -1
        perfect_rot = np.dot(U, Vt)
        
    matrix[:3, :3] = perfect_rot
    return matrix

def get_low_poly_proxy(mesh, target_faces=1000):
    """
    Dynamically decimates mesh using a safe, dependency-free library.
    """
    if len(mesh.faces) <= target_faces:
        return mesh
        
    try:
        # Fast, safe decimation that won't break your numpy versions
        vertices, faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=target_faces
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    except Exception as e:
        return mesh

def load_assembly_from_folder(folder_path, bounding_box_type="AABB"):
    """
    Loads assembly meshes (.obj or .stl) and computes bounding box properties.
    
    bounding_box_type: 'OBB' (Oriented, uses PCA to find axes) 
                    or 'AABB' (Axis-Aligned, uses global CAD axes)
    """
    assembly_manifest = {}
    matrix_idx = 0
    
    folder = Path(folder_path)
    
    # 1. Look for both .obj and .stl files (case-insensitive)
    mesh_files = sorted(
        list(folder.glob("*.obj")) + list(folder.glob("*.OBJ")) +
        list(folder.glob("*.stl")) + list(folder.glob("*.STL"))
    )

    if not mesh_files:
        raise FileNotFoundError(f"No .obj or .stl files found in {folder_path}")

    for file_path in mesh_files:
        # Use the exact file stem as the primary key so it matches Fabrica's part identifiers
        exact_stem = file_path.stem 
        
        # Load mesh via trimesh (works natively for both .obj and .stl)
        mesh_geom = trimesh.load(str(file_path), force='mesh')
        
        # Ensure it's a valid Trimesh object
        if isinstance(mesh_geom, trimesh.Scene):
            mesh_geom = mesh_geom.dump(concatenate=True)
            
        mesh_geom.merge_vertices()
        
        # ---> FORCE TRIMESH TO BUILD THE CACHE ONCE <---
        _ = mesh_geom.face_adjacency
        
        if bounding_box_type == "OBB":
            # Get the Oriented Bounding Box transformation matrix
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh_geom)
            
            # Clean the matrix and re-orthogonalize it
            to_origin = clean_obb_matrix(to_origin)
            from_origin = np.linalg.inv(to_origin)
            
            v_x = from_origin[:3, 0]
            v_y = from_origin[:3, 1]
            v_z = from_origin[:3, 2]
            
        elif bounding_box_type == "AABB":
            # Pure Translation (No Rotation, perfectly aligned to CAD axes)
            center = mesh_geom.bounding_box.centroid
            
            to_origin = np.eye(4) 
            to_origin[:3, 3] = -center
            from_origin = np.linalg.inv(to_origin)
            
            # Global cardinal axes
            v_x = np.array([1.0, 0.0, 0.0])
            v_y = np.array([0.0, 1.0, 0.0])
            v_z = np.array([0.0, 0.0, 1.0])
            
        else:
            raise ValueError("bounding_box_type must be either 'OBB' or 'AABB'")

        extraction_vectors = {
            "+x": v_x, "-x": -v_x,
            "+y": v_y, "-y": -v_y,
            "+z": v_z, "-z": -v_z
        }

        assembly_manifest[exact_stem] = {
            "matrix_idx": matrix_idx,
            "part_mesh": mesh_geom,
            "file_path": str(file_path),
            "to_origin": to_origin,             
            "extraction_vectors": extraction_vectors,
            "center_point": from_origin[:3, 3],
            "face_adjacency": mesh_geom.face_adjacency.copy(),
            "triangles": mesh_geom.triangles.copy(),       # <--- CACHE RAW ARRAYS
            "face_normals": mesh_geom.face_normals.copy()  # <--- CACHE RAW ARRAYS
        }
        
        matrix_idx += 1
        
    return assembly_manifest

def load_fabrica_assembly_from_folder(obj_dir, part_ids, bounding_box_type="AABB"):
    """
    Loads assembly meshes based EXACTLY on Fabrica's sorted part_ids.
    """
    assembly_manifest = {}
    
    # Iterate through Fabrica's list. enumerate() automatically assigns 
    # the exact matrix_idx that corresponds to Fabrica's part order.
    for matrix_idx, part_id in enumerate(part_ids):
        
        # Reconstruct the exact file path
        file_path = os.path.join(obj_dir, part_id + '.obj')
        
        # Load mesh
        mesh_geom = trimesh.load(file_path, force='mesh')
        if isinstance(mesh_geom, trimesh.Scene):
            mesh_geom = mesh_geom.dump(concatenate=True)
        mesh_geom.merge_vertices()
        
        if bounding_box_type == "OBB":
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh_geom)
            to_origin = clean_obb_matrix(to_origin)
            from_origin = np.linalg.inv(to_origin)
            v_x, v_y, v_z = from_origin[:3, 0], from_origin[:3, 1], from_origin[:3, 2]
            
        elif bounding_box_type == "AABB":
            center = mesh_geom.bounding_box.centroid
            to_origin = np.eye(4) 
            to_origin[:3, 3] = -center
            from_origin = np.linalg.inv(to_origin)
            v_x = np.array([1.0, 0.0, 0.0])
            v_y = np.array([0.0, 1.0, 0.0])
            v_z = np.array([0.0, 0.0, 1.0])
            
        else:
            raise ValueError("bounding_box_type must be either 'OBB' or 'AABB'")

        assembly_manifest[part_id] = {
            "matrix_idx": matrix_idx,
            "part_mesh": mesh_geom,
            "file_path": file_path,
            "to_origin": to_origin,             
            "extraction_vectors": {"+x": v_x, "-x": -v_x, "+y": v_y, "-y": -v_y, "+z": v_z, "-z": -v_z},
            "center_point": from_origin[:3, 3],
            "face_adjacency": mesh_geom.face_adjacency.copy(), # <--- ADD THIS
            "triangles": mesh_geom.triangles.copy(),           # <--- ADD THIS
            "face_normals": mesh_geom.face_normals.copy()      # <--- ADD THIS
        }
        
    return assembly_manifest

def export_matrices_to_excel(matrices, assembly_manifest, output_folder="output_matrices", filename="Interference_Matrices.xlsx"):
    """
    Exports the 6 directional matrices to a single Excel file, 
    putting each matrix on its own named tab.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)
    
    # Extract part names sorted by their matrix_idx to guarantee exact alignment
    sorted_parts = sorted(assembly_manifest.items(), key=lambda x: x[1]["matrix_idx"])
    part_names = [name for name, data in sorted_parts]

    # Open the Excel writer engine
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for direction, matrix in matrices.items():
            
            # Create the DataFrame
            df = pd.DataFrame(matrix, index=part_names, columns=part_names)
            df.index.name = "Moving \ Stationary"
            
            # Create a clean tab name (e.g., "+x" becomes "Pos_X")
            tab_name = direction.replace("+", "Pos_").replace("-", "Neg_").upper()
            
            # Write this specific matrix to its own tab
            df.to_excel(writer, sheet_name=tab_name)
            
    print(f"Successfully saved all matrices to a single Excel file: {filepath}")

def export_matrices_to_csv(matrices, assembly_manifest, output_folder="output_matrices"):
    """
    Exports the 6 directional matrices to CSV files with part names as headers.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract part names sorted by their matrix_idx to guarantee exact alignment
    sorted_parts = sorted(assembly_manifest.items(), key=lambda x: x[1]["matrix_idx"])
    part_names = [name for name, data in sorted_parts]

    # Export each direction as its own spreadsheet
    for direction, matrix in matrices.items():
        # Create a Pandas DataFrame to bind the matrix to the part names
        df = pd.DataFrame(matrix, index=part_names, columns=part_names)
        
        # Save it cleanly to disk
        safe_dir_name = direction.replace("+", "Pos_").replace("-", "Neg_")
        filepath = os.path.join(output_folder, f"IM_Matrix_{safe_dir_name}.csv")
        
        df.to_csv(filepath)
        print(f"Successfully saved {direction} matrix to: {filepath}")             

def export_directions_to_excel(assembly_manifest, output_folder="output_matrices", filename="Robot_Extraction_Vectors.xlsx"):
    """
    Exports the 3D extraction vectors for each part to an Excel sheet.
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)
    
    # Prepare the data dictionary for Pandas
    data = {"Part Name": []}
    directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
    for d in directions:
        data[d] = []
        
    for part_name, properties in assembly_manifest.items():
        data["Part Name"].append(part_name)
        for d in directions:
            # Format the vector as a clean, rounded string: "[1.000, 0.000, 0.000]"
            v = properties["extraction_vectors"][d]
            data[d].append(f"[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
            
    # Export to Excel
    df = pd.DataFrame(data)
    df.set_index("Part Name", inplace=True)
    df.to_excel(filepath)
    print(f"Successfully saved Robot Extraction Vectors to: {filepath}")

## Auxiliary visualization Functions with pyvista
def visualize_pseudofaces(part, pseudo_faces_list, plotter, color_idx, show = False):
    """
    Renders the full part in transparent gray, and paints each 
    Pseudo Face object a different solid color.
    """
    # 1. Convert the trimesh object to a pyvista object
    mesh = pv.wrap(part)
    
    # 2. Set up the 3D window
    pl = plotter
    
    # 3. Draw the original part as a faint "ghost" for context
    pl.add_mesh(mesh, color='white', opacity=0.15)
    
    # 4. A list of bright colors to cycle through
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange']
    
    # 5. Loop through your instantiated PseudoFace objects
    for i, pf in enumerate(pseudo_faces_list):
        # Extract only the triangles that belong to this PF!
        # Because we converted face_indices to a numpy array, this works instantly.
        pf_mesh = mesh.extract_cells(pf.face_indices)
        
        # Pick a color (loops back to the start if you have more than 7 PFs)
        # c = colors[i % len(colors)]
        c = colors[i % len(colors)]
        
        # Draw this specific PF solid and show its black triangle edges
        pl.add_mesh(pf_mesh, color=c, show_edges=True, line_width=1)
        
    # Show the interactive window!
    if show:
        pl.show()

def visualize_part_axes(part_name, assembly_manifest):
    """
    Visualizes a specific part in its original global position 
    and draws its extraction axes based on its OBB.
    """

    # X axis: red
    # Y axis: green
    # Z axis: blue
    properties = assembly_manifest[part_name]
    mesh = properties["part_mesh"]
    center = properties["center_point"]
    vectors = properties["extraction_vectors"]
    
    plotter = pv.Plotter(title=f"Extraction Axes: {part_name}")
    plotter.add_mesh(pv.wrap(mesh), color="lightgray", opacity=0.8, show_edges=True)

    # Add an arrow for the Local +X axis (Red)
    arrow_x = pv.Arrow(start=center, direction=vectors["+x"], scale=15)
    plotter.add_mesh(arrow_x, color='red')

    # Add an arrow for the Local +Y axis (Green)
    arrow_y = pv.Arrow(start=center, direction=vectors["+y"], scale=15)
    plotter.add_mesh(arrow_y, color='green')

    # Add an arrow for the Local +Z axis (Blue)
    arrow_z = pv.Arrow(start=center, direction=vectors["+z"], scale=15)
    plotter.add_mesh(arrow_z, color='blue')

    # Add a tiny sphere at the center point so we can see the origin of the arrows
    plotter.add_mesh(pv.Sphere(radius=1.5, center=center), color='black')

    plotter.show()

def visualize_narrow_phase(pseudo_faces, overlap_region, plotter, index, show = False):
    
    for i, pf in enumerate(pseudo_faces):
        if i == 0:
            pf.visualize_focus_facets(overlap_region, plotter, index)
        else:
            pf.visualize_focus_facets(overlap_region, plotter, index, show_SR_box=False)
    if show:
        plotter.show()
# ----------------------------------------------------- COMPLEMENTARY FUNCTIONS ---------------------------------------------- 

# For the loop i need to revert the transformation applied to part_b so i
# can get extraction directions in the original frame
if __name__ == "__main__":
    # Change this whenever you test a new assembly
    input_folder = 'STLs/EndEffector2'
    
    # Extract the assembly name dynamically (e.g., "EndEffector")
    assembly_name = Path(input_folder).name
    
    # Define dynamic output paths
    out_obb = f"Outputs_{assembly_name}/OBB"
    out_aabb = f"Outputs_{assembly_name}/AABB"
    
    # =======================================================
    #                 RUN 1: OBB PIPELINE
    # =======================================================
    # print(f"\n[STARTING] OBB Pipeline for: {assembly_name}...")
    # assembly_manifest_OBB = load_assembly_from_folder(input_folder, bounding_box_type="OBB")
    
    # start_time = time.time()
    # final_matrices_OBB = calculate_IM_matrices(assembly_manifest_OBB)
    # print(f"--- OBB Math Complete! Time Taken: {(time.time() - start_time):.2f} seconds ---")
    
    # export_matrices_to_excel(final_matrices_OBB, assembly_manifest_OBB, output_folder=out_obb)
    # export_directions_to_excel(assembly_manifest_OBB, output_folder=out_obb)


    # =======================================================
    #                 RUN 2: AABB PIPELINE
    # =======================================================
    print(f"\n[STARTING] AABB Pipeline for: {assembly_name}...")
    assembly_manifest_AABB = load_assembly_from_folder(input_folder, bounding_box_type="AABB")
    #visualize_part_axes('Hook_v2-2', assembly_manifest_AABB)
    print(assembly_manifest_AABB.keys())

    # Test the optimized action row generation for a specific part
    optimized_vector = np.array([1.0, 0.0, 0.0])
    test_part = 'Hook_v2-2'
    test_row = get_optimized_action_row(test_part, assembly_manifest_AABB, optimized_vector)
    print(f'Optimized action row for {test_part}: {test_row}')
    
    # start_time = time.time()
    # final_matrices_AABB = calculate_IM_matrices(assembly_manifest_AABB)
    # print(f"--- AABB Math Complete! Time Taken: {(time.time() - start_time):.2f} seconds ---")
    
    # export_matrices_to_excel(final_matrices_AABB, assembly_manifest_AABB, output_folder=out_aabb)
    # export_directions_to_excel(assembly_manifest_AABB, output_folder=out_aabb)

    # print(f"\n>>> All pipelines finished! Check the '{out_obb}' and '{out_aabb}' folders.")




