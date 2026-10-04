import os, shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--assembly-dir', type=str, required=True)
parser.add_argument('--log-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

for subdir_name in os.listdir(args.assembly_dir):
    sub_log_dir = os.path.join(args.log_dir, subdir_name)
    sub_assembly_dir = os.path.join(args.assembly_dir, subdir_name)
    if os.path.isdir(sub_assembly_dir):
        sub_output_dir = os.path.join(args.output_dir, subdir_name)
        os.makedirs(sub_output_dir, exist_ok=True)
        for obj_name in os.listdir(sub_assembly_dir):
            obj_path = os.path.join(sub_assembly_dir, obj_name)
            if obj_path.endswith('.obj'):
                shutil.copy(obj_path, os.path.join(sub_output_dir, f'{obj_name}.obj'))
        sub_fixture_dir = os.path.join(sub_log_dir, 'fixture')
        shutil.copytree(sub_fixture_dir, os.path.join(sub_output_dir, 'fixture'))
        os.remove(os.path.join(sub_output_dir, 'fixture', 'pickup.json'))
    print(f'Done: {subdir_name}')
