import os
import shutil
import json


def get_files_to_copy(stage):
    files = []
    files += ['precedence.pkl', 'precedence.png']
    if stage == 'precedence':
        return files
    files += ['grasps.pkl', 'grasp_stats.txt']
    if stage == 'grasp':
        return files
    files += ['tree.pkl', 'tree.png']
    if stage == 'seq_plan':
        return files
    files += ['tree_opt.pkl', 'tree_opt.png']
    if stage == 'seq_opt':
        return files
    files += ['fixture']
    if stage == 'fixture':
        return files
    files += ['motion.pkl']
    if stage == 'motion':
        return files
    raise ValueError('Invalid stage')


def get_stats_keys_to_copy(stage):
    keys = []
    keys += ['preced_plan']
    if stage == 'precedence':
        return keys
    keys += ['grasp_gen']
    if stage == 'grasp':
        return keys
    keys += ['seq_plan']
    if stage == 'seq_plan':
        return keys
    keys += ['seq_opt']
    if stage == 'seq_opt':
        return keys
    keys += ['fixture_gen']
    if stage == 'fixture':
        return keys
    keys += ['motion_plan']
    if stage == 'motion':
        return keys
    raise ValueError('Invalid stage')


def copy_logs(source_dir, target_dir, files, stats_keys):
    for file in files:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        if not os.path.exists(source_path):
            print(f'Warning: {source_path} does not exist')
            continue
        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copyfile(source_path, target_path)
    with open(os.path.join(source_dir, 'stats.json'), 'r') as f:
        stats = json.load(f)
    stats = {key: stats[key] for key in stats_keys if key in stats}
    with open(os.path.join(target_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f)


def copy_logs_batch(source_dir, target_dir, files, stats_keys):
    for subdir in os.listdir(source_dir):
        source_subdir = os.path.join(source_dir, subdir)
        target_subdir = os.path.join(target_dir, subdir)
        if os.path.isdir(source_subdir):
            os.makedirs(target_subdir, exist_ok=True)
            copy_logs(source_subdir, target_subdir, files, stats_keys)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--source-dir', type=str, required=True)
    parser.add_argument('--target-dir', type=str, required=True)
    parser.add_argument('--stage', type=str, required=True, choices=['precedence', 'grasp', 'seq_plan', 'seq_opt', 'fixture', 'motion'])
    args = parser.parse_args()
    source_dir, target_dir, stage = args.source_dir, args.target_dir, args.stage

    files = get_files_to_copy(stage)
    stats_keys = get_stats_keys_to_copy(stage)
    copy_logs_batch(source_dir, target_dir, files, stats_keys)
