import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import trimesh

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)


def prepare_parts_printing(assembly_dir, printing_dir):

    os.makedirs(printing_dir, exist_ok=True)

    for subdir_name in os.listdir(assembly_dir):
        subdir = os.path.join(assembly_dir, subdir_name)
        if os.path.isdir(subdir):
            printing_subdir = os.path.join(printing_dir, subdir_name)
            os.makedirs(printing_subdir, exist_ok=True)
            for part_file in os.listdir(subdir):
                part_path = os.path.join(subdir, part_file)
                if part_path.endswith('.obj'):
                    mesh = trimesh.load(part_path)
                    mesh.apply_scale(10)
                    part_output_path = os.path.join(printing_subdir, part_file.replace('.obj', '.stl'))
                    mesh.export(part_output_path)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--printing-dir', type=str, required=True)
    args = parser.parse_args()

    prepare_parts_printing(args.assembly_dir, args.printing_dir)
    