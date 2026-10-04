import os
import json

def update_json_stats(stats_path, key, value):
    """
    Reads existing stats (if any), updates the given key, and writes
    the file such that each top-level key is on its own single line inside {}.
    """
    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as fp:
                stats = json.load(fp)
        except (json.JSONDecodeError, OSError):
            stats = {}

    stats[key] = value

    # Write formatted with each top-level entry on its own line
    with open(stats_path, 'w') as fp:
        fp.write('{\n')
        lines = [f'  {json.dumps(k)}: {json.dumps(v)}' for k, v in stats.items()]
        fp.write(',\n'.join(lines))
        fp.write('\n}\n')