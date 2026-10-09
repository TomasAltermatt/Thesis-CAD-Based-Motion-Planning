import os
import sys
import subprocess
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Sequentially run all result analysis and plotting scripts.")
    parser.add_argument('test_name', type=str, help="Name of the test directory in logs/")
    args = parser.parse_args()

    # The directory where this runner script is located
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    # Defined in strict order to satisfy data dependencies
    scripts_to_run = [
        "analyze_insertion_kinematics.py", # Extracts data to .pkl
        "plot_insertion_kinematics.py",    # Requires the .pkl from above
        "full_joint_trajectories.py",
        "get_log_comparison.py",
        "get_move_hold_scores.py",
        "plot_manipulability.py"
    ]

    print(f"=== Starting Batch Results Processing for: {args.test_name} ===")
    total_start_time = time.time()

    for script_name in scripts_to_run:
        script_path = os.path.join(scripts_dir, script_name)
        
        # Verify the script exists (in case you used the _2.py suffixes)
        if not os.path.exists(script_path):
            print(f"\n[WARNING] Could not find {script_name}. Skipping...")
            continue

        print(f"\n---> Running: {script_name}")
        try:
            # sys.executable ensures it uses the same Python environment you are currently in
            result = subprocess.run(
                [sys.executable, script_path, args.test_name], 
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {script_name} failed with exit code {e.returncode}.")
            print("Batch execution halted to prevent cascading errors.")
            sys.exit(1)

    elapsed_time = round(time.time() - total_start_time, 2)
    print(f"\n=== Batch Processing Complete in {elapsed_time}s ===")
    print(f"All outputs safely routed to results/{args.test_name}/")

if __name__ == '__main__':
    main()