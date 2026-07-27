import torch


def get_curriculum_difficulty(insertion_success, difficulty_curr, difficulty_delta=0.1):
    if insertion_success > 0.8:
        difficulty_new = min(max(difficulty_curr + difficulty_delta, 0.0), 1.0)
    elif insertion_success < 0.5:
        difficulty_new = min(max(difficulty_curr - difficulty_delta, 0.0), 1.0)
    else:
        difficulty_new = difficulty_curr
    return difficulty_new


@torch.jit.script
def sample_random_points_on_trajectory(trajectory):
    """
    Samples a random point strictly on the trajectory with linear interpolation.
    
    Args:
        trajectory (torch.Tensor): Tensor of shape [N, T, 3] representing N trajectories of length T.
    
    Returns:
        torch.Tensor: Sampled points of shape [N, 3]
    """
    N, T, _ = trajectory.shape
    
    # Randomly sample fractional time indices within [0, T-1]
    random_time_indices = torch.rand(N, dtype=trajectory.dtype, device=trajectory.device) * (T - 1)

    # Get the integer indices for interpolation
    lower_indices = torch.floor(random_time_indices).long()
    upper_indices = torch.ceil(random_time_indices).long()
    
    # Ensure upper index does not exceed T-1
    upper_indices = torch.clamp(upper_indices, max=T-1)

    # Compute interpolation weights
    alpha = random_time_indices - lower_indices.float()
    
    # Get corresponding trajectory points
    lower_points = trajectory[torch.arange(N), lower_indices]  # Shape [N, 3]
    upper_points = trajectory[torch.arange(N), upper_indices]  # Shape [N, 3]
    
    # Perform linear interpolation
    sampled_points = (1 - alpha[:, None]) * lower_points + alpha[:, None] * upper_points
    
    return sampled_points


@torch.jit.script
def closest_point_on_path(path_points: torch.Tensor, query_points: torch.Tensor, threshold: float):
    """
    Finds the closest points on a batch of piecewise-continuous paths to a batch of query points.
    Optionally adjusts query points to the closest point on the path if the distance exceeds a threshold.

    Args:
        path_points (torch.Tensor): Tensor of shape (N, L, 3), where N is the batch size,
                                   and L is the number of points in each path.
        query_points (torch.Tensor): Tensor of shape (N, 3) representing the batch of query points.
        threshold (float): Distance threshold. Query points farther than
                                     the threshold are adjusted to be within the threshold.

    Returns:
        closest_points (torch.Tensor): Tensor of shape (N, 3) containing the closest points on the paths.
        min_distances (torch.Tensor): Tensor of shape (N,) containing the distances to the closest points.
        adjusted_query_points (torch.Tensor): Tensor of shape (N, 3) containing the adjusted query points
    """
    N, L, _ = path_points.shape

    # Compute vectors between consecutive points in the path
    segments = path_points[:, 1:] - path_points[:, :-1]  # Shape: (N, L-1, 3)

    # Vectors from the first point of each segment to the query points
    query_vectors = query_points.unsqueeze(1) - path_points[:, :-1]  # Shape: (N, L-1, 3)

    # Segment lengths squared
    segment_lengths_squared = torch.sum(segments ** 2, dim=-1, keepdim=True)  # Shape: (N, L-1, 1)

    # Avoid division by zero for degenerate segments
    segment_lengths_squared = torch.where(segment_lengths_squared == 0, torch.ones_like(segment_lengths_squared), segment_lengths_squared)

    # Projection factors onto the segments
    projection_factors = torch.sum(query_vectors * segments, dim=-1, keepdim=True) / segment_lengths_squared  # Shape: (N, L-1, 1)

    # Clamp the projection factors to [0, 1] to stay within the segments
    projection_factors = torch.clamp(projection_factors, 0, 1)  # Shape: (N, L-1, 1)

    # Compute the closest points on the segments
    projected_points = path_points[:, :-1] + projection_factors * segments  # Shape: (N, L-1, 3)

    # Compute distances to the query points
    distances = torch.norm(projected_points - query_points.unsqueeze(1), dim=-1)  # Shape: (N, L-1)

    # Find the minimum distances and their indices
    min_distances, min_indices = torch.min(distances, dim=-1)  # Shape: (N,)

    # Gather the closest points using the indices
    closest_points = projected_points[torch.arange(N), min_indices]  # Shape: (N, 3)

    # Adjust query points that are farther than the threshold
    exceeded_mask = min_distances > threshold  # Shape: (N,)
    adjusted_query_points = query_points.clone()
    adjusted_query_points[exceeded_mask] = (
        closest_points[exceeded_mask] + 
        (query_points[exceeded_mask] - closest_points[exceeded_mask]).renorm(p=2, dim=0, maxnorm=threshold)
    )

    return closest_points, min_distances, adjusted_query_points


@torch.jit.script
def do_deltapos_path_transform(delta_position: torch.Tensor, path_start: torch.Tensor, path_end: torch.Tensor) -> torch.Tensor:
    path_vector = path_end - path_start  # Compute direction from end to start
    path_direction = path_vector / torch.norm(path_vector, dim=1, keepdim=True)

    target_direction = torch.tensor([0, 0, -1], dtype=delta_position.dtype, device=delta_position.device).unsqueeze(0).expand_as(path_direction)

    v = torch.cross(path_direction, target_direction, dim=1)  # Cross product
    c = torch.sum(path_direction * target_direction, dim=1, keepdim=True).unsqueeze(-1)  # Shape [N, 1, 1]
    s = torch.norm(v, dim=1, keepdim=True).unsqueeze(-1)  # Shape [N, 1, 1]

    eye = torch.eye(3, dtype=delta_position.dtype, device=delta_position.device).unsqueeze(0).repeat(delta_position.shape[0], 1, 1)
    v_cross = torch.zeros_like(eye)

    v_cross[:, 0, 1] = -v[:, 2]
    v_cross[:, 0, 2] = v[:, 1]
    v_cross[:, 1, 0] = v[:, 2]
    v_cross[:, 1, 2] = -v[:, 0]
    v_cross[:, 2, 0] = -v[:, 1]
    v_cross[:, 2, 1] = v[:, 0]

    factor = (1 - c) / (s ** 2 + 1e-8)
    factor = factor.expand(-1, 3, 3)  # Ensure correct shape for broadcasting

    R = eye + v_cross + torch.bmm(v_cross, v_cross) * factor

    delta_transformed = torch.bmm(R, delta_position.unsqueeze(-1)).squeeze(-1)

    return delta_transformed


@torch.jit.script
def undo_deltapos_path_transform(delta_transformed: torch.Tensor, path_start: torch.Tensor, path_end: torch.Tensor) -> torch.Tensor:
    path_vector = path_end - path_start
    path_direction = path_vector / torch.norm(path_vector, dim=1, keepdim=True)

    target_direction = torch.tensor([0, 0, -1], dtype=delta_transformed.dtype, device=delta_transformed.device).unsqueeze(0).expand_as(path_direction)

    v = torch.cross(target_direction, path_direction, dim=1)  # Cross product
    c = torch.sum(target_direction * path_direction, dim=1, keepdim=True).unsqueeze(-1)  # Shape [N, 1, 1]
    s = torch.norm(v, dim=1, keepdim=True).unsqueeze(-1)  # Shape [N, 1, 1]

    eye = torch.eye(3, dtype=delta_transformed.dtype, device=delta_transformed.device).unsqueeze(0).repeat(delta_transformed.shape[0], 1, 1)
    v_cross = torch.zeros_like(eye)

    v_cross[:, 0, 1] = -v[:, 2]
    v_cross[:, 0, 2] = v[:, 1]
    v_cross[:, 1, 0] = v[:, 2]
    v_cross[:, 1, 2] = -v[:, 0]
    v_cross[:, 2, 0] = -v[:, 1]
    v_cross[:, 2, 1] = v[:, 0]

    factor = (1 - c) / (s ** 2 + 1e-8)
    factor = factor.expand(-1, 3, 3)  # Ensure correct shape for broadcasting

    R_inv = eye + v_cross + torch.bmm(v_cross, v_cross) * factor

    delta_world = torch.bmm(R_inv, delta_transformed.unsqueeze(-1)).squeeze(-1)

    return delta_world


@torch.jit.script
def do_pos_path_transform(position: torch.Tensor, path_start: torch.Tensor, path_end: torch.Tensor) -> torch.Tensor:
    position_relative = position - path_start
    return do_deltapos_path_transform(position_relative, path_start, path_end)


@torch.jit.script
def undo_pos_path_transform(position_transformed: torch.Tensor, path_start: torch.Tensor, path_end: torch.Tensor) -> torch.Tensor:
    position_relative = undo_deltapos_path_transform(position_transformed, path_start, path_end)
    return position_relative + path_start
