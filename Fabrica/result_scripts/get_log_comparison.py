import os
import json
import csv
import argparse
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_base_dir)

def main(test_name):
    base_dir = os.path.join('logs', test_name)
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        return

    assemblies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    phases = ['preced_plan', 'grasp_gen', 'seq_plan', 'seq_opt', 'fixture_gen', 'motion_plan']
    
    csv_data = []
    header = ['Assembly'] + [f"{phase.replace('_', ' ').title()} (Orig/New)" for phase in phases]
    csv_data.append(header)

    for assembly in assemblies:
        row = [assembly]
        orig_path = os.path.join(base_dir, assembly, 'original', 'stats.json')
        new_path = os.path.join(base_dir, assembly, 'new', 'stats.json')
        
        orig_stats, new_stats = {}, {}
        if os.path.exists(orig_path):
            with open(orig_path, 'r') as f: orig_stats = json.load(f)
        if os.path.exists(new_path):
            with open(new_path, 'r') as f: new_stats = json.load(f)
                
        for phase in phases:
            orig_time = orig_stats.get(phase, {}).get('time', 'N/A')
            new_time = new_stats.get(phase, {}).get('time', 'N/A')
            row.append(f"{orig_time} / {new_time}")
            
        csv_data.append(row)

    # --- ROUTED TO CSV DIRECTORY ---
    results_dir = os.path.join('results', test_name, 'csv_reports')
    os.makedirs(results_dir, exist_ok=True)
    output_filepath = os.path.join(results_dir, 'time_comparisons.csv')
    
    with open(output_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    print(f"Successfully exported time comparisons to {output_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_name', type=str, help='Name of the test directory')
    args = parser.parse_args()
    main(args.test_name)