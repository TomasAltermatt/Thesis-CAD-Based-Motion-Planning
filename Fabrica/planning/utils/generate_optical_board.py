import trimesh
import trimesh.transformations as tf


def generate_hole_cylinder(hole_diameter, hole_height, segments=32):
    '''
    Generates a cylindrical hole
    Parameters:
        hole_diameter (float): Diameter of the cylindrical hole
        hole_height (float): Height of the cylindrical hole
        segments (int): Number of segments for approximating the circular base
    '''
    return trimesh.creation.cylinder(radius=hole_diameter / 2, height=hole_height, sections=segments, transform=tf.translation_matrix([0, 0, -hole_height / 2]))


def generate_optical_board(board_dim=(36, 24), hole_gap=2.5, hole_diameter=0.5, board_thickness=1.0):
    '''
    Generates an optical board with holes
    Parameters:
        board_dim (tuple): Number of holes in x and y directions
        hole_gap (float): Distance between holes
        hole_diameter (float): Diameter of the cylindrical hole
        board_thickness (float): Thickness of the optical board
    '''
    board_box = trimesh.creation.box([(board_dim[0] + 1) * hole_gap, (board_dim[1] + 1) * hole_gap, board_thickness])
    board_box.apply_translation([0, 0, -board_thickness / 2]) # board upper surface at z=0, center at origin
    lower_left_hole_center = (-(board_dim[0] - 1) / 2 * hole_gap, -(board_dim[1] - 1) / 2 * hole_gap, 0)
    hole_cylinders = []
    for i in range(board_dim[0]):
        for j in range(board_dim[1]):
            hole_center = [lower_left_hole_center[0] + i * hole_gap, lower_left_hole_center[1] + j * hole_gap, lower_left_hole_center[2]]
            hole_cylinder = generate_hole_cylinder(hole_diameter, board_thickness, segments=32)
            hole_cylinder.apply_translation(hole_center)
            hole_cylinders.append(hole_cylinder)
    return trimesh.boolean.difference([board_box, trimesh.util.concatenate(hole_cylinders)])


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--render', default=False, action='store_true')
    args = parser.parse_args()

    mesh = generate_optical_board()
    if args.output_path is not None:
        mesh.export(args.output_path)
    if args.render:
        mesh.show()
