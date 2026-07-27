import re
import argparse

# Function to extract assembly and insertion success
def extract_assembly_success(log_file_path):
    assembly_success_dict = {}
    
    with open(log_file_path, 'r') as file:
        log_content = file.read()
        
        # Regex patterns
        assembly_pattern = re.compile(r'Running evaluation for assembly: (\w+)')
        success_pattern = re.compile(r'Insertion Success: ([0-9.]+)')
        
        # Find all matches
        assemblies = assembly_pattern.findall(log_content)
        successes = success_pattern.findall(log_content)

        n_success_per_assembly = len(successes) // len(assemblies)
        successes = successes[::n_success_per_assembly]
        
        # Ensure paired extraction, only keep the first success for each assembly
        for assembly, success in zip(assemblies, successes):
            assembly_success_dict[assembly] = float(success)
    
    return assembly_success_dict

# Main function to handle argument parsing
def main():
    parser = argparse.ArgumentParser(description='Extract assembly and insertion success from log file.')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    args = parser.parse_args()

    pairs = extract_assembly_success(args.log_file)
    
    # Desired order
    order = [
        'beam', 'plumbers_block', 'car', 'gamepad', 
        'cooling_manifold', 'duct', 'stool_circular'
    ]

    # Format output in LaTeX style
    formatted_output = ''
    for assembly in order:
        success = pairs.get(assembly, 0.0) * 100  # Default to 0% if missing
        formatted_output += f'{success:.2f}\% & '
    
    formatted_output = formatted_output.rstrip(' & ') + ' \\\\'
    print(formatted_output)

if __name__ == '__main__':
    main()
