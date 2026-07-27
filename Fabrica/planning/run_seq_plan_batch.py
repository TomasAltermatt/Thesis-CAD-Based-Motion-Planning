import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from planning.run_seq_plan import run_seq_plan
from utils.parallel import parallel_execute


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--assembly-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--num-proc', type=int, default=1)
    parser.add_argument('--early-term', default=False, action='store_true')
    parser.add_argument('--plot', default=False, action='store_true')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    worker_args = []
    for subdir_name in os.listdir(args.assembly_dir):
        sub_assembly_dir = os.path.join(args.assembly_dir, subdir_name)
        if os.path.isdir(sub_assembly_dir):
            sub_log_dir = os.path.join(args.log_dir, subdir_name)
            worker_args.append((sub_assembly_dir, sub_log_dir, args.early_term, args.plot))
    try:
        for _ in parallel_execute(run_seq_plan, worker_args, num_proc=args.num_proc):
            pass
    except KeyboardInterrupt:
        exit()
