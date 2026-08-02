# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

cdef inline bint check_2d_sat_overlap(double[:, :] t1, double[:, :] t2):
    """
    Pure C implementation of the Separating Axis Theorem (SAT) for 2D triangles.
    Replaces Shapely's poly.intersects() completely.
    """
    cdef int i, j, k
    cdef double nx, ny, dot_val
    cdef double min1, max1, min2, max2
    
    # Test all 3 edges of Triangle 1
    for i in range(3):
        j = (i + 1) % 3
        nx = t1[j][1] - t1[i][1]
        ny = t1[i][0] - t1[j][0]
        
        min1 = max1 = t1[0][0]*nx + t1[0][1]*ny
        for k in range(1, 3):
            dot_val = t1[k][0]*nx + t1[k][1]*ny
            if dot_val < min1: min1 = dot_val
            if dot_val > max1: max1 = dot_val
            
        min2 = max2 = t2[0][0]*nx + t2[0][1]*ny
        for k in range(1, 3):
            dot_val = t2[k][0]*nx + t2[k][1]*ny
            if dot_val < min2: min2 = dot_val
            if dot_val > max2: max2 = dot_val
            
        if max1 < min2 or max2 < min1:
            return False
            
    # Test all 3 edges of Triangle 2
    for i in range(3):
        j = (i + 1) % 3
        nx = t2[j][1] - t2[i][1]
        ny = t2[i][0] - t2[j][0]
        
        min1 = max1 = t1[0][0]*nx + t1[0][1]*ny
        for k in range(1, 3):
            dot_val = t1[k][0]*nx + t1[k][1]*ny
            if dot_val < min1: min1 = dot_val
            if dot_val > max1: max1 = dot_val
            
        min2 = max2 = t2[0][0]*nx + t2[0][1]*ny
        for k in range(1, 3):
            dot_val = t2[k][0]*nx + t2[k][1]*ny
            if dot_val < min2: min2 = dot_val
            if dot_val > max2: max2 = dot_val
            
        if max1 < min2 or max2 < min1:
            return False
            
    return True

cpdef list get_intersecting_pairs_c(double[:, :, :] tris_a_2d, double[:, :, :] tris_b_2d):
    """
    Takes two arrays of 2D triangles, runs AABB and SAT checks in C, 
    and returns a clean Python list of index pairs that truly intersect.
    """
    cdef int num_a = tris_a_2d.shape[0]
    cdef int num_b = tris_b_2d.shape[0]
    cdef int i, j
    cdef list intersecting_pairs = []
    
    # Fast C-level bounding box variables
    cdef double min_a_x, max_a_x, min_a_y, max_a_y
    cdef double min_b_x, max_b_x, min_b_y, max_b_y
    cdef int k
    
    for i in range(num_a):
        min_a_x = max_a_x = tris_a_2d[i][0][0]
        min_a_y = max_a_y = tris_a_2d[i][0][1]
        for k in range(1, 3):
            if tris_a_2d[i][k][0] < min_a_x: min_a_x = tris_a_2d[i][k][0]
            if tris_a_2d[i][k][0] > max_a_x: max_a_x = tris_a_2d[i][k][0]
            if tris_a_2d[i][k][1] < min_a_y: min_a_y = tris_a_2d[i][k][1]
            if tris_a_2d[i][k][1] > max_a_y: max_a_y = tris_a_2d[i][k][1]
            
        for j in range(num_b):
            min_b_x = max_b_x = tris_b_2d[j][0][0]
            min_b_y = max_b_y = tris_b_2d[j][0][1]
            for k in range(1, 3):
                if tris_b_2d[j][k][0] < min_b_x: min_b_x = tris_b_2d[j][k][0]
                if tris_b_2d[j][k][0] > max_b_x: max_b_x = tris_b_2d[j][k][0]
                if tris_b_2d[j][k][1] < min_b_y: min_b_y = tris_b_2d[j][k][1]
                if tris_b_2d[j][k][1] > max_b_y: max_b_y = tris_b_2d[j][k][1]
                
            # C-Speed AABB Filter
            if max_a_x < min_b_x or min_a_x > max_b_x or max_a_y < min_b_y or min_a_y > max_b_y:
                continue
                
            # C-Speed SAT Math Filter
            if check_2d_sat_overlap(tris_a_2d[i], tris_b_2d[j]):
                intersecting_pairs.append((i, j))
                
    return intersecting_pairs


cpdef bint fast_any_intersection_c(double[:, :, :] tris_a_2d, double[:, :, :] tris_b_2d):
    """
    Early-exit SAT overlap check. 
    Returns True instantly upon finding a single intersection.
    """
    cdef int num_a = tris_a_2d.shape[0]
    cdef int num_b = tris_b_2d.shape[0]
    cdef int i, j, k
    
    cdef double min_a_x, max_a_x, min_a_y, max_a_y
    cdef double min_b_x, max_b_x, min_b_y, max_b_y
    
    for i in range(num_a):
        min_a_x = max_a_x = tris_a_2d[i][0][0]
        min_a_y = max_a_y = tris_a_2d[i][0][1]
        for k in range(1, 3):
            if tris_a_2d[i][k][0] < min_a_x: min_a_x = tris_a_2d[i][k][0]
            if tris_a_2d[i][k][0] > max_a_x: max_a_x = tris_a_2d[i][k][0]
            if tris_a_2d[i][k][1] < min_a_y: min_a_y = tris_a_2d[i][k][1]
            if tris_a_2d[i][k][1] > max_a_y: max_a_y = tris_a_2d[i][k][1]
            
        for j in range(num_b):
            min_b_x = max_b_x = tris_b_2d[j][0][0]
            min_b_y = max_b_y = tris_b_2d[j][0][1]
            for k in range(1, 3):
                if tris_b_2d[j][k][0] < min_b_x: min_b_x = tris_b_2d[j][k][0]
                if tris_b_2d[j][k][0] > max_b_x: max_b_x = tris_b_2d[j][k][0]
                if tris_b_2d[j][k][1] < min_b_y: min_b_y = tris_b_2d[j][k][1]
                if tris_b_2d[j][k][1] > max_b_y: max_b_y = tris_b_2d[j][k][1]
                
            # AABB Filter
            if max_a_x < min_b_x or min_a_x > max_b_x or max_a_y < min_b_y or min_a_y > max_b_y:
                continue
                
            # SAT Filter - Instant Exit
            if check_2d_sat_overlap(tris_a_2d[i], tris_b_2d[j]):
                return True
                
    return False


# ---------------------------------------------------------
# PURE C TRIANGLE CLIPPING & KINEMATIC DEPTH EVALUATION
# ---------------------------------------------------------

cdef inline bint point_in_triangle(double px, double py, double[:, :] t):
    """Barycentric test to see if a vertex is inside a triangle."""
    cdef double d1 = (px - t[0][0]) * (t[1][1] - t[0][1]) - (py - t[0][1]) * (t[1][0] - t[0][0])
    cdef double d2 = (px - t[1][0]) * (t[2][1] - t[1][1]) - (py - t[1][1]) * (t[2][0] - t[1][0])
    cdef double d3 = (px - t[2][0]) * (t[0][1] - t[2][1]) - (py - t[2][1]) * (t[0][0] - t[2][0])
    
    # 1e-7 tolerance absorbs CAD floating-point noise for perfectly flush edges
    cdef bint has_neg = (d1 < -1e-7) or (d2 < -1e-7) or (d3 < -1e-7)
    cdef bint has_pos = (d1 > 1e-7)  or (d2 > 1e-7)  or (d3 > 1e-7)
    
    return not (has_neg and has_pos)

cdef inline bint get_line_intersection(double p0_x, double p0_y, double p1_x, double p1_y,
                               double p2_x, double p2_y, double p3_x, double p3_y,
                               double* out_x, double* out_y):
    """Calculates exactly where two segment lines cross."""
    cdef double s1_x = p1_x - p0_x
    cdef double s1_y = p1_y - p0_y
    cdef double s2_x = p3_x - p2_x
    cdef double s2_y = p3_y - p2_y
    
    cdef double denom = -s2_x * s1_y + s1_x * s2_y
    if denom >= -1e-8 and denom <= 1e-8:
        return False # Parallel or collinear
        
    cdef double s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / denom
    cdef double t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denom
    
    # If the crossing point happens strictly within the length of both segments
    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        out_x[0] = p0_x + (t * s1_x)
        out_y[0] = p0_y + (t * s1_y)
        return True
        
    return False

cpdef tuple evaluate_deep_narrow_phase_c(
    double[:, :, :] tris_a_2d, double[:, :, :] tris_b_2d,
    double[:, :, :] tris_a_3d, double[:, :, :] tris_b_3d,
    double[:, :] normals_a, double[:, :] normals_b,
    list intersecting_pairs,
    int w_idx, int u_idx, int v_idx,
    double w_tol, double n_tol,
    bint use_MRT, double mrt_tol
):
    cdef int max_pos = 0
    cdef int max_neg = 0
    cdef int pair_idx, a_idx, b_idx, i, j, p
    cdef int p_count
    cdef int interference_type
    
    # Stack-allocated variables
    cdef double overlap_pts[20][2] 
    cdef double ix, iy
    cdef double min_u_overlap, max_u_overlap, min_v_overlap, max_v_overlap
    cdef double overlap_dist_u, overlap_dist_v, overlap_distance
    
    cdef double nu_a, nv_a, nw_a, d_a
    cdef double nu_b, nv_b, nw_b, d_b
    cdef double proj_w_a, proj_w_b
    
    cdef int num_pairs = len(intersecting_pairs)
    
    for pair_idx in range(num_pairs):
        a_idx = intersecting_pairs[pair_idx][0]
        b_idx = intersecting_pairs[pair_idx][1]
        
        p_count = 0
        
        # 1. Grab vertices of A inside B
        for i in range(3):
            if point_in_triangle(tris_a_2d[a_idx][i][0], tris_a_2d[a_idx][i][1], tris_b_2d[b_idx]):
                overlap_pts[p_count][0] = tris_a_2d[a_idx][i][0]
                overlap_pts[p_count][1] = tris_a_2d[a_idx][i][1]
                p_count += 1
        
        # 2. Grab vertices of B inside A
        for i in range(3):
            if point_in_triangle(tris_b_2d[b_idx][i][0], tris_b_2d[b_idx][i][1], tris_a_2d[a_idx]):
                overlap_pts[p_count][0] = tris_b_2d[b_idx][i][0]
                overlap_pts[p_count][1] = tris_b_2d[b_idx][i][1]
                p_count += 1
                
        # 3. Grab precise Edge Intersections
        for i in range(3):
            for j in range(3):
                if get_line_intersection(
                    tris_a_2d[a_idx][i][0], tris_a_2d[a_idx][i][1], tris_a_2d[a_idx][(i+1)%3][0], tris_a_2d[a_idx][(i+1)%3][1],
                    tris_b_2d[b_idx][j][0], tris_b_2d[b_idx][j][1], tris_b_2d[b_idx][(j+1)%3][0], tris_b_2d[b_idx][(j+1)%3][1],
                    &ix, &iy
                ):
                    overlap_pts[p_count][0] = ix
                    overlap_pts[p_count][1] = iy
                    p_count += 1
                    if p_count == 20: break 
            if p_count == 20: break
                    
        if p_count == 0: continue
            
        # ---> NATIVE C MRT OVERLAP DISTANCE CALCULATION <---
        min_u_overlap = overlap_pts[0][0]
        max_u_overlap = overlap_pts[0][0]
        min_v_overlap = overlap_pts[0][1]
        max_v_overlap = overlap_pts[0][1]
        
        for p in range(1, p_count):
            if overlap_pts[p][0] < min_u_overlap: min_u_overlap = overlap_pts[p][0]
            if overlap_pts[p][0] > max_u_overlap: max_u_overlap = overlap_pts[p][0]
            if overlap_pts[p][1] < min_v_overlap: min_v_overlap = overlap_pts[p][1]
            if overlap_pts[p][1] > max_v_overlap: max_v_overlap = overlap_pts[p][1]
            
        overlap_dist_u = max_u_overlap - min_u_overlap
        overlap_dist_v = max_v_overlap - min_v_overlap
        overlap_distance = overlap_dist_u if overlap_dist_u < overlap_dist_v else overlap_dist_v
        
        interference_type = 2
        if use_MRT and overlap_distance < mrt_tol:
            interference_type = 1
        # -----------------------------------------------------

        # 4. Run the depth math!
        nu_a = normals_a[a_idx][u_idx]
        nv_a = normals_a[a_idx][v_idx]
        nw_a = normals_a[a_idx][w_idx]
        d_a = -(nu_a * tris_a_3d[a_idx][0][u_idx] + nv_a * tris_a_3d[a_idx][0][v_idx] + nw_a * tris_a_3d[a_idx][0][w_idx])
        
        nu_b = normals_b[b_idx][u_idx]
        nv_b = normals_b[b_idx][v_idx]
        nw_b = normals_b[b_idx][w_idx]
        d_b = -(nu_b * tris_b_3d[b_idx][0][u_idx] + nv_b * tris_b_3d[b_idx][0][v_idx] + nw_b * tris_b_3d[b_idx][0][w_idx])
        
        for p in range(p_count):
            if nw_a > -1e-6 and nw_a < 1e-6:
                proj_w_a = tris_a_3d[a_idx][0][w_idx]
            else:
                proj_w_a = -(nu_a * overlap_pts[p][0] + nv_a * overlap_pts[p][1] + d_a) / nw_a
                
            if nw_b > -1e-6 and nw_b < 1e-6:
                proj_w_b = tris_b_3d[b_idx][0][w_idx]
            else:
                proj_w_b = -(nu_b * overlap_pts[p][0] + nv_b * overlap_pts[p][1] + d_b) / nw_b
                
            if nw_a > n_tol and nw_b < -n_tol:
                if proj_w_a <= proj_w_b + w_tol:
                    if interference_type > max_pos: max_pos = interference_type
                    
            elif nw_a < -n_tol and nw_b > n_tol:
                if proj_w_a >= proj_w_b - w_tol:
                    if interference_type > max_neg: max_neg = interference_type
                    
            if max_pos == 2 and max_neg == 2:
                return (2, 2)
                
    return (max_pos, max_neg)