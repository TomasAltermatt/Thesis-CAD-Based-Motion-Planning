import numpy as np


def closest_point_on_path(path_points, query_point, threshold):
    """
    Finds the closest point on a piecewise-continuous path to a query point.
    Optionally adjusts the query point to the closest point within a given threshold.

    Args:
        path_points (np.ndarray): Array of shape (L, 3), where L is the number of points in the path.
        query_point (np.ndarray): Array of shape (3,) representing the query point.
        threshold (float, optional): Distance threshold. If specified, query point farther than
                                     the threshold is adjusted to be within the threshold.

    Returns:
        closest_point (np.ndarray): The closest point on the path.
        min_distance (float): The distance to the closest point.
        adjusted_query_point (np.ndarray): The adjusted query point if threshold is provided, otherwise None.
    """
    L = path_points.shape[0]

    # Compute vectors between consecutive points in the path
    segments = path_points[1:] - path_points[:-1]  # Shape: (L-1, 3)

    # Vectors from the first point of each segment to the query point
    query_vectors = query_point - path_points[:-1]  # Shape: (L-1, 3)

    # Segment lengths squared
    segment_lengths_squared = np.sum(segments ** 2, axis=-1, keepdims=True)  # Shape: (L-1, 1)
    segment_lengths_squared = np.where(segment_lengths_squared == 0, 1.0, segment_lengths_squared)

    # Projection factors onto the segments
    projection_factors = np.sum(query_vectors * segments, axis=-1, keepdims=True) / segment_lengths_squared

    # Clamp the projection factors to [0, 1] to stay within the segments
    projection_factors = np.clip(projection_factors, 0, 1)

    # Compute the closest points on the segments
    projected_points = path_points[:-1] + projection_factors * segments  # Shape: (L-1, 3)

    # Compute distances to the query point
    distances = np.linalg.norm(projected_points - query_point, axis=-1)

    # Find the minimum distance and its index
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    closest_point = projected_points[min_index]

    adjusted_query_point = None
    direction = query_point - closest_point
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm * threshold
    adjusted_query_point = closest_point + direction

    return closest_point, min_distance, adjusted_query_point


def do_deltapos_path_transform(delta_position, path_start, path_end):
    path_vector = path_end - path_start
    path_direction = path_vector / np.linalg.norm(path_vector)
    
    target_direction = np.array([0, 0, -1])
    
    v = np.cross(path_direction, target_direction)
    c = np.dot(path_direction, target_direction)
    s = np.linalg.norm(v)
    
    eye = np.eye(3)
    v_cross = np.zeros_like(eye)
    
    v_cross[0, 1] = -v[2]
    v_cross[0, 2] = v[1]
    v_cross[1, 0] = v[2]
    v_cross[1, 2] = -v[0]
    v_cross[2, 0] = -v[1]
    v_cross[2, 1] = v[0]
    
    factor = (1 - c) / (s ** 2 + 1e-8)
    
    R = eye + v_cross + np.matmul(v_cross, v_cross) * factor
    
    delta_transformed = np.dot(R, delta_position)
    
    return delta_transformed


def undo_deltapos_path_transform(delta_transformed, path_start, path_end):
    path_vector = path_end - path_start
    path_direction = path_vector / np.linalg.norm(path_vector)
    
    target_direction = np.array([0, 0, -1])
    
    v = np.cross(target_direction, path_direction)
    c = np.dot(target_direction, path_direction)
    s = np.linalg.norm(v)
    
    eye = np.eye(3)
    v_cross = np.zeros_like(eye)
    
    v_cross[0, 1] = -v[2]
    v_cross[0, 2] = v[1]
    v_cross[1, 0] = v[2]
    v_cross[1, 2] = -v[0]
    v_cross[2, 0] = -v[1]
    v_cross[2, 1] = v[0]
    
    factor = (1 - c) / (s ** 2 + 1e-8)
    
    R_inv = eye + v_cross + np.matmul(v_cross, v_cross) * factor
    
    delta_world = np.dot(R_inv, delta_transformed)
    
    return delta_world
