import os
import trimesh
from tabulate import tabulate
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--log-dir', type=str, required=True)
args = parser.parse_args()

table = []

for subdir_name in os.listdir(args.log_dir):
    sub_log_dir = os.path.join(args.log_dir, subdir_name)
    fixture_path = os.path.join(sub_log_dir, 'fixture', 'fixture.obj')
    if os.path.exists(fixture_path):
        mesh = trimesh.load_mesh(fixture_path)
        table.append([subdir_name, mesh.extents[0], mesh.extents[1], mesh.extents[2]])

print(tabulate(table, headers=['assembly', 'x', 'y', 'z'], tablefmt='grid'))
