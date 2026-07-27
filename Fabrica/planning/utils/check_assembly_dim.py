import os
import trimesh
from tabulate import tabulate
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--assembly-dir', type=str, required=True)
args = parser.parse_args()

table = []

for subdir_name in os.listdir(args.assembly_dir):
    sub_assembly_dir = os.path.join(args.assembly_dir, subdir_name)
    if os.path.isdir(sub_assembly_dir):
        meshes = []
        for filename in os.listdir(sub_assembly_dir):
            if filename.endswith('.obj'):
                obj_path = os.path.join(sub_assembly_dir, filename)
                mesh = trimesh.load_mesh(obj_path)
                meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
        table.append([subdir_name, mesh.extents[0], mesh.extents[1], mesh.extents[2]])

print(tabulate(table, headers=['assembly', 'x', 'y', 'z'], tablefmt='grid'))
