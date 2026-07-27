import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

from learning.preprocessing.prepare_isaac_plan_info import prepare_isaac_plan_info
from utils.parallel import parallel_execute


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--plan-info-dir', type=str, required=True)
    parser.add_argument('--num-proc', type=int, default=1)
    args = parser.parse_args()

    worker_args = []
    for subdir_name in os.listdir(args.log_dir):
        sub_log_dir = os.path.join(args.log_dir, subdir_name)
        if os.path.isdir(sub_log_dir):
            sub_plan_info_path = os.path.join(args.plan_info_dir, subdir_name + '.pkl')
            worker_args.append((sub_log_dir, sub_plan_info_path))

    try:
        for _ in parallel_execute(prepare_isaac_plan_info, worker_args, num_proc=args.num_proc):
            pass
    except KeyboardInterrupt:
        exit()

