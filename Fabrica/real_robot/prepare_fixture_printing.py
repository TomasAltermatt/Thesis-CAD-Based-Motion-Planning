import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import trimesh

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)


def prepare_fixture_printing(log_dir, printing_dir):

    os.makedirs(printing_dir, exist_ok=True)

    for subdir_name in os.listdir(log_dir):
        subdir = os.path.join(log_dir, subdir_name)
        if os.path.isdir(subdir):
            fixture_input_path = os.path.join(subdir, 'fixture', 'fixture.obj')
            if not os.path.exists(fixture_input_path):
                print(f'Fixture not found for {subdir_name}, skipping')
                continue

            mesh = trimesh.load(fixture_input_path)
            mesh.apply_scale(10)
            extents = mesh.extents
            assert extents[0] <= 250.0 and extents[2] <= 250.0 and extents[1] <= 500.0
            if extents[1] <= 250.0:
                fixture_output_path = os.path.join(printing_dir, subdir_name + '.stl')
                mesh.export(fixture_output_path)
            else:
                mesh1 = mesh.slice_plane([0, mesh.centroid[1], 0], [0, 1, 0], cap=True)
                mesh2 = mesh.slice_plane([0, mesh.centroid[1], 0], [0, -1, 0], cap=True)
                fixture_output_path1 = os.path.join(printing_dir, subdir_name + '_1.stl')
                fixture_output_path2 = os.path.join(printing_dir, subdir_name + '_2.stl')
                mesh1.export(fixture_output_path1)
                mesh2.export(fixture_output_path2)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--printing-dir', type=str, required=True)
    args = parser.parse_args()

    prepare_fixture_printing(args.log_dir, args.printing_dir)
    