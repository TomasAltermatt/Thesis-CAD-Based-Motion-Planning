import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from rendering.render_motion_plan import render_motion_plan, get_camera_option
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
    parser.add_argument('--make-gif', default=False, action='store_true')
    parser.add_argument('--camera-option', type=int, default=1)
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None)
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None)
    parser.add_argument('--num-proc', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    if args.record_dir is not None: os.makedirs(args.record_dir, exist_ok=True)

    camera_lookat, camera_pos = get_camera_option(args.camera_option)
    if args.camera_lookat is not None:
        camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        camera_pos = args.camera_pos

    worker_args = []
    for subdir_name in os.listdir(args.assembly_dir):
        sub_log_dir = os.path.join(args.log_dir, subdir_name)
        sub_assembly_dir = os.path.join(args.assembly_dir, subdir_name)
        if os.path.isdir(sub_assembly_dir):
            record_path = os.path.join(args.record_dir, f'{subdir_name}.gif' if args.make_gif else f'{subdir_name}.mp4') if args.record_dir is not None else None
            worker_args.append((sub_assembly_dir, sub_log_dir, record_path, args.make_gif, camera_lookat, camera_pos, None, args.seed, args.fps))
    try:
        for _ in parallel_execute(render_motion_plan, worker_args, num_proc=args.num_proc):
            pass
    except KeyboardInterrupt:
        exit()
