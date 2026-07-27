import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from assets.make_sdf import make_sdf
from utils.parallel import parallel_execute


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--num-proc', type=int, default=8)
    args = parser.parse_args()

    worker_args = []
    for assembly_id in os.listdir(args.dir):
        assembly_dir = os.path.join(args.dir, assembly_id)
        if os.path.isdir(assembly_dir):
            worker_args.append([assembly_dir, args.sdf_dx])

    for _ in parallel_execute(make_sdf, worker_args, num_proc=args.num_proc):
        pass
