import os
import json
from time import time
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='multi_assembly', help='directory storing all assemblies')
    parser.add_argument('--id-path', type=str, default=None)
    parser.add_argument('--steps', type=int, default=1, help='number of simulation steps')
    parser.add_argument('--non-fixed', default=False, action='store_true', help='whether parts can move')
    parser.add_argument('--body-type', type=str, default='bvh')
    parser.add_argument('--sdf-dx', type=float, default=10, help='grid resolution of SDF')
    parser.add_argument('--label-path', type=str, default=None)
    parser.add_argument('--ground-z', type=float, default=-5)

    args = parser.parse_args()
    if args.id_path is None:
        ids = [x for x in os.listdir(args.dir) if not (x.startswith('.') or x.startswith('_') or x.endswith('.json'))]
        ids = sorted(ids, key=lambda x: int(x))
    else:
        with open(args.id_path, 'r') as fp:
            if args.id_path.endswith('.txt'):
                ids = fp.read().splitlines()
            elif args.id_path.endswith('.json'):
                ids = list(json.load(fp).keys())
            else:
                raise Exception

    i = 0

    t_start = time()

    if args.label_path is not None:
        labels = {}
    else:
        labels = None

    while i < len(ids):
        print(ids[i], f' [{i}/{len(ids)}] T = {int(time() - t_start)}s')
        cmd = f'python test_multi_sim.py --id {ids[i]} --dir {args.dir} ' \
            f'--steps {args.steps} --body-type {args.body_type} --sdf-dx {args.sdf_dx} --ground-z {args.ground_z} '
        if not args.non_fixed: cmd += '--fixed '
        os.system(cmd)

        label = input()

        if labels is not None:
            labels[ids[i]] = label

            with open(args.label_path, 'w') as fp:
                fp.writelines([f'{k}: {v}\n' for k, v in labels.items()])

        if label == 'p':
            print('go back to the previous one')
            i = max(i - 1, 0)
        else:
            i += 1