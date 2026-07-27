import os
import json
from tabulate import tabulate
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--log-dir', type=str, required=True)
args = parser.parse_args()

steps = ['preced_plan', 'grasp_gen', 'seq_plan', 'seq_opt', 'fixture_gen', 'motion_plan']
table = []

for subdir_name in os.listdir(args.log_dir):
    sub_log_dir = os.path.join(args.log_dir, subdir_name)
    stats_path = os.path.join(sub_log_dir, 'stats.json')
    if os.path.exists(stats_path):
        row = [subdir_name]
        with open(stats_path, 'r') as fp:
            stats = json.load(fp)
        for step in steps:
            row.append(stats[step]['time'])
        row.append(sum(row[1:]))
        table.append(row)

print(tabulate(table, headers=['assembly'] + steps + ['total'], tablefmt='grid'))
