# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

# ---> ::1 FORCES C-CONTIGUOUS POINTER ARITHMETIC (Massive Speedup) <---
cdef inline bint check_2d_sat_overlap(double[:, :, ::1] t1, int i, double[:, :, ::1] t2, int j) nogil:
    cdef int e, e_next, k
    cdef double nx, ny, dot_val
    cdef double min1, max1, min2, max2
    
    for e in range(3):
        # ---> ELIMINATED SLOW MODULO (%) MATH <---
        e_next = e + 1
        if e_next == 3: e_next = 0
        
        nx = t1[i, e_next, 1] - t1[i, e, 1]
        ny = t1[i, e, 0] - t1[i, e_next, 0]
        
        min1 = max1 = t1[i, 0, 0]*nx + t1[i, 0, 1]*ny
        for k in range(1, 3):
            dot_val = t1[i, k, 0]*nx + t1[i, k, 1]*ny
            if dot_val < min1: min1 = dot_val
            if dot_val > max1: max1 = dot_val
            
        min2 = max2 = t2[j, 0, 0]*nx + t2[j, 0, 1]*ny
        for k in range(1, 3):
            dot_val = t2[j, k, 0]*nx + t2[j, k, 1]*ny
            if dot_val < min2: min2 = dot_val
            if dot_val > max2: max2 = dot_val
            
        if max1 < min2 or max2 < min1:
            return False
            
    for e in range(3):
        e_next = e + 1
        if e_next == 3: e_next = 0
        
        nx = t2[j, e_next, 1] - t2[j, e, 1]
        ny = t2[j, e, 0] - t2[j, e_next, 0]
        
        min1 = max1 = t1[i, 0, 0]*nx + t1[i, 0, 1]*ny
        for k in range(1, 3):
            dot_val = t1[i, k, 0]*nx + t1[i, k, 1]*ny
            if dot_val < min1: min1 = dot_val
            if dot_val > max1: max1 = dot_val
            
        min2 = max2 = t2[j, 0, 0]*nx + t2[j, 0, 1]*ny
        for k in range(1, 3):
            dot_val = t2[j, k, 0]*nx + t2[j, k, 1]*ny
            if dot_val < min2: min2 = dot_val
            if dot_val > max2: max2 = dot_val
            
        if max1 < min2 or max2 < min1:
            return False
            
    return True

cpdef bint fast_any_intersection_c(double[:, :, ::1] tris_a_2d, double[:, :, ::1] tris_b_2d, double[:, ::1] aabbs_a, double[:, ::1] aabbs_b):
    cdef int num_a = tris_a_2d.shape[0]
    cdef int num_b = tris_b_2d.shape[0]
    cdef int i, j
    cdef int start_j = 0
    
    if num_a == 0 or num_b == 0:
        return False
        
    cdef double a_min_x, a_max_x, a_min_y, a_max_y
    
    for i in range(num_a):
        a_min_x = aabbs_a[i, 0]
        a_min_y = aabbs_a[i, 1]
        a_max_x = aabbs_a[i, 2]
        a_max_y = aabbs_a[i, 3]
        
        while start_j < num_b and aabbs_b[start_j, 2] < a_min_x:
            start_j += 1
        
        for j in range(start_j, num_b):
            if aabbs_b[j, 0] > a_max_x:
                break
            
            if aabbs_b[j, 2] < a_min_x:
                continue
            
            if aabbs_b[j, 1] > a_max_y or aabbs_b[j, 3] < a_min_y:
                continue
                
            if check_2d_sat_overlap(tris_a_2d, i, tris_b_2d, j):
                return True
                
    return False


cdef inline bint point_in_triangle(double px, double py, double[:, :, ::1] t, int idx) nogil:
    cdef double d1 = (px - t[idx, 0, 0]) * (t[idx, 1, 1] - t[idx, 0, 1]) - (py - t[idx, 0, 1]) * (t[idx, 1, 0] - t[idx, 0, 0])
    cdef double d2 = (px - t[idx, 1, 0]) * (t[idx, 2, 1] - t[idx, 1, 1]) - (py - t[idx, 1, 1]) * (t[idx, 2, 0] - t[idx, 1, 0])
    cdef double d3 = (px - t[idx, 2, 0]) * (t[idx, 0, 1] - t[idx, 2, 1]) - (py - t[idx, 2, 1]) * (t[idx, 0, 0] - t[idx, 2, 0])
    
    cdef bint has_neg = (d1 < -1e-7) or (d2 < -1e-7) or (d3 < -1e-7)
    cdef bint has_pos = (d1 > 1e-7)  or (d2 > 1e-7)  or (d3 > 1e-7)
    
    return not (has_neg and has_pos)

cdef inline bint get_line_intersection(double p0_x, double p0_y, double p1_x, double p1_y,
                               double p2_x, double p2_y, double p3_x, double p3_y,
                               double* out_x, double* out_y) nogil:
    cdef double s1_x = p1_x - p0_x
    cdef double s1_y = p1_y - p0_y
    cdef double s2_x = p3_x - p2_x
    cdef double s2_y = p3_y - p2_y
    
    cdef double denom = -s2_x * s1_y + s1_x * s2_y
    if denom >= -1e-8 and denom <= 1e-8:
        return False
        
    cdef double s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / denom
    cdef double t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denom
    
    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        out_x[0] = p0_x + (t * s1_x)
        out_y[0] = p0_y + (t * s1_y)
        return True
        
    return False


cpdef tuple evaluate_overlap_c(
    double[:, :, ::1] tris_a_2d, double[:, :, ::1] tris_b_2d,
    double[:, ::1] aabbs_a, double[:, ::1] aabbs_b,
    double[:, :, ::1] tris_a_3d, double[:, :, ::1] tris_b_3d,
    double[:, ::1] normals_a, double[:, ::1] normals_b,
    int w_idx, int u_idx, int v_idx,
    double w_tol, double n_tol,
    bint use_MRT, double mrt_tol,
    int abort_threshold
):
    cdef int num_a = tris_a_2d.shape[0]
    cdef int num_b = tris_b_2d.shape[0]
    cdef int i, j, p, k, k1, k2, k1_next, k2_next
    cdef int start_j = 0
    cdef int max_pos = 0
    cdef int max_neg = 0
    cdef int p_count = 0
    cdef int interference_type
    cdef int pair_count = 0
    
    cdef double a_min_x, a_max_x, a_min_y, a_max_y
    cdef double overlap_pts[20][2] 
    cdef double ix, iy
    cdef double min_u_overlap, max_u_overlap, min_v_overlap, max_v_overlap
    cdef double overlap_dist_u, overlap_dist_v, overlap_distance
    cdef double nu_a, nv_a, nw_a, d_a
    cdef double nu_b, nv_b, nw_b, d_b
    cdef double proj_w_a, proj_w_b
    
    if num_a == 0 or num_b == 0:
        return (0, 0)
        
    for i in range(num_a):
        a_min_x = aabbs_a[i, 0]
        a_min_y = aabbs_a[i, 1]
        a_max_x = aabbs_a[i, 2]
        a_max_y = aabbs_a[i, 3]
        
        while start_j < num_b and aabbs_b[start_j, 2] < a_min_x:
            start_j += 1
            
        for j in range(start_j, num_b):
            if aabbs_b[j, 0] > a_max_x:
                break 
                
            if aabbs_b[j, 2] < a_min_x:
                continue
                
            if aabbs_b[j, 1] > a_max_y or aabbs_b[j, 3] < a_min_y:
                continue
                
            if check_2d_sat_overlap(tris_a_2d, i, tris_b_2d, j):
                pair_count += 1
                if abort_threshold > 0 and pair_count > abort_threshold:
                    return (-999, -999)
                    
                p_count = 0
                for k in range(3):
                    if point_in_triangle(tris_a_2d[i, k, 0], tris_a_2d[i, k, 1], tris_b_2d, j):
                        overlap_pts[p_count][0] = tris_a_2d[i, k, 0]
                        overlap_pts[p_count][1] = tris_a_2d[i, k, 1]
                        p_count += 1
                
                for k in range(3):
                    if point_in_triangle(tris_b_2d[j, k, 0], tris_b_2d[j, k, 1], tris_a_2d, i):
                        overlap_pts[p_count][0] = tris_b_2d[j, k, 0]
                        overlap_pts[p_count][1] = tris_b_2d[j, k, 1]
                        p_count += 1
                        
                for k1 in range(3):
                    k1_next = k1 + 1
                    if k1_next == 3: k1_next = 0
                    for k2 in range(3):
                        k2_next = k2 + 1
                        if k2_next == 3: k2_next = 0
                        
                        if get_line_intersection(
                            tris_a_2d[i, k1, 0], tris_a_2d[i, k1, 1], tris_a_2d[i, k1_next, 0], tris_a_2d[i, k1_next, 1],
                            tris_b_2d[j, k2, 0], tris_b_2d[j, k2, 1], tris_b_2d[j, k2_next, 0], tris_b_2d[j, k2_next, 1],
                            &ix, &iy
                        ):
                            overlap_pts[p_count][0] = ix
                            overlap_pts[p_count][1] = iy
                            p_count += 1
                            if p_count == 20: break 
                    if p_count == 20: break
                            
                if p_count == 0: continue
                
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

                nu_a = normals_a[i, u_idx]
                nv_a = normals_a[i, v_idx]
                nw_a = normals_a[i, w_idx]
                d_a = -(nu_a * tris_a_3d[i, 0, u_idx] + nv_a * tris_a_3d[i, 0, v_idx] + nw_a * tris_a_3d[i, 0, w_idx])
                
                nu_b = normals_b[j, u_idx]
                nv_b = normals_b[j, v_idx]
                nw_b = normals_b[j, w_idx]
                d_b = -(nu_b * tris_b_3d[j, 0, u_idx] + nv_b * tris_b_3d[j, 0, v_idx] + nw_b * tris_b_3d[j, 0, w_idx])
                
                for p in range(p_count):
                    if nw_a > -1e-6 and nw_a < 1e-6:
                        proj_w_a = tris_a_3d[i, 0, w_idx]
                    else:
                        proj_w_a = -(nu_a * overlap_pts[p][0] + nv_a * overlap_pts[p][1] + d_a) / nw_a
                        
                    if nw_b > -1e-6 and nw_b < 1e-6:
                        proj_w_b = tris_b_3d[j, 0, w_idx]
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