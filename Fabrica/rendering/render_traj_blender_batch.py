import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from rendering.render_traj_blender import render_traj
from utils.parallel import parallel_execute


if __name__ == '__main__':
    import platform, multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--record-dir', type=str, default=None)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--keep-img', default=False, action='store_true')
    parser.add_argument('--num-proc', type=int, default=1)
    args = parser.parse_args()

    if args.record_dir is not None: os.makedirs(args.record_dir, exist_ok=True)

    worker_args = []
    for subdir_name in os.listdir(args.assembly_dir):
        sub_log_dir = os.path.join(args.log_dir, subdir_name)
        sub_assembly_dir = os.path.join(args.assembly_dir, subdir_name)
        if os.path.isdir(sub_assembly_dir):
            record_path = os.path.join(args.record_dir, f'{subdir_name}.mp4') if args.record_dir is not None else None
            worker_args.append((sub_assembly_dir, sub_log_dir, record_path, args.interval, args.keep_img))
    try:
        for _ in parallel_execute(render_traj, worker_args, num_proc=args.num_proc):
            pass
    except KeyboardInterrupt:
        exit()
