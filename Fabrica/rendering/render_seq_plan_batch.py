import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from rendering.render_seq_plan import render_seq_plan, get_camera_option
from utils.parallel import parallel_execute


if __name__ == '__main__':
    import platform, multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--optimized', default=False, action='store_true')
    parser.add_argument('--record-dir', type=str, required=True)
    parser.add_argument('--reverse', default=False, action='store_true')
    parser.add_argument('--combine-record', default=False, action='store_true')
    parser.add_argument('--make-gif', default=False, action='store_true')
    parser.add_argument('--camera-option', type=int, default=1)
    parser.add_argument('--camera-lookat', type=float, nargs=3, default=None)
    parser.add_argument('--camera-pos', type=float, nargs=3, default=None)
    parser.add_argument('--num-proc', type=int, default=1)
    parser.add_argument('--inner-num-proc', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    camera_lookat, camera_pos = get_camera_option(args.camera_option)
    if args.camera_lookat is not None:
        camera_lookat = args.camera_lookat
    if args.camera_pos is not None:
        camera_pos = args.camera_pos

    os.makedirs(args.record_dir, exist_ok=True)

    worker_args = []
    for subdir_name in os.listdir(args.assembly_dir):
        sub_log_dir = os.path.join(args.log_dir, subdir_name)
        sub_assembly_dir = os.path.join(args.assembly_dir, subdir_name)
        if os.path.isdir(sub_assembly_dir):
            sub_record_dir = os.path.join(args.record_dir, subdir_name)
            worker_args.append((sub_assembly_dir, sub_log_dir, args.optimized, sub_record_dir, args.reverse, args.combine_record, args.make_gif, camera_lookat, camera_pos, args.inner_num_proc, args.seed, args.fps))
    try:
        for _ in parallel_execute(render_seq_plan, worker_args, num_proc=args.num_proc):
            pass
    except KeyboardInterrupt:
        exit()
