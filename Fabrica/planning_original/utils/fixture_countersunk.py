import numpy as np
import trimesh


COUNTERSUNK_DIAMETER = 1.05
HOLE_DIAMETER = 0.6
PAD_DIAMETER = 2.5
PAD_HEIGHT = 0.5


def create_solid_cone(radius, height, segments=32):
    """
    Creates a watertight cone with a bottom cap in trimesh.

    Parameters:
        radius (float): Radius of the cone's base.
        height (float): Height of the cone.
        segments (int): Number of segments for approximating the circular base.

    Returns:
        trimesh.Trimesh: A watertight cone mesh with a bottom cap.
    """
    # Create the cone (open at the base)
    cone = trimesh.creation.cone(radius=radius, height=height, sections=segments)

    # Create a disk for the base cap
    theta = np.linspace(0, 2 * np.pi, segments)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(x, -height / 2)  # Base is at z = -height/2

    # Vertices of the disk
    disk_vertices = np.vstack((x, y, z)).T

    # Add the center point of the disk
    center_vertex = np.array([0, 0, -height / 2])  # Base center
    disk_vertices = np.vstack((disk_vertices, center_vertex))

    # Create faces for the disk
    disk_faces = []
    for i in range(segments - 1):
        disk_faces.append([i, i + 1, len(disk_vertices) - 1])  # Triangle fan
    disk_faces.append([segments - 1, 0, len(disk_vertices) - 1])  # Last triangle

    disk_faces = np.array(disk_faces)

    # Combine cone and disk
    vertices = np.vstack((cone.vertices, disk_vertices))
    faces = np.vstack((cone.faces, disk_faces + len(cone.vertices)))

    # Create a new mesh
    solid_cone = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Ensure the mesh is watertight
    solid_cone.merge_vertices()
    solid_cone.fix_normals()

    solid_cone = solid_cone.convex_hull   # Convex hull to ensure watertightness

    # Return the resulting watertight cone mesh
    return solid_cone


def generate_countersunk_hole(countersunk_diameter, hole_diameter, pad_height, segments=32):
    '''
    Generates a countersunk hole with a cylindrical hole and a cone on top
    Parameters:
        countersunk_diameter (float): Diameter of the countersunk hole
        hole_diameter (float): Diameter of the cylindrical hole
        pad_height (float): Height of the pad to drill the hole into
        segments (int): Number of segments for approximating the circular base
    '''
    countersunk_cone_height = countersunk_diameter / 2 - hole_diameter / 2
    hole_height = pad_height - countersunk_cone_height
    countersunk_cone = create_solid_cone(countersunk_diameter / 2, countersunk_diameter / 2, segments=segments)
    countersunk_cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    countersunk_cone.apply_transform(trimesh.transformations.translation_matrix([0, 0, hole_height + countersunk_cone_height]))
    hole_cylinder = trimesh.creation.cylinder(radius=hole_diameter / 2, height=hole_height, sections=segments, transform=trimesh.transformations.translation_matrix([0, 0, hole_height / 2]))
    mesh = trimesh.boolean.union([hole_cylinder, countersunk_cone])
    return mesh


def generate_countersunk_pad(countersunk_diameter=COUNTERSUNK_DIAMETER, hole_diameter=HOLE_DIAMETER, pad_diameter=PAD_DIAMETER, pad_height=PAD_HEIGHT):
    '''
    Generates a countersunk pad with a hole in the middle
    '''
    pad_board = trimesh.creation.box([pad_diameter, pad_diameter, pad_height], transform=trimesh.transformations.translation_matrix([0, 0, pad_height / 2]))
    hole = generate_countersunk_hole(countersunk_diameter, hole_diameter, pad_height)
    return trimesh.boolean.difference([pad_board, hole])


if __name__ == '__main__':
    mesh = generate_countersunk_pad()
    mesh.show()
