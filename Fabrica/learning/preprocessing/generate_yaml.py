import os
import yaml

# Base path where 'fabrica' is located
base_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fabrica', 'mesh', 'fabrica')
yaml_output_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fabrica', 'yaml', 'fabrica_asset_info', 'fabrica.yaml')

# Dictionary to hold all the data
asset_info = {}

# Walk through the directory structure
for root, dirs, files in os.walk(base_path):
    # Get the relative path from base_path to root
    relative_path = os.path.relpath(root, base_path)
    
    # If we are in a subfolder directly under 'fabrica'
    if relative_path != '.' and not relative_path.startswith('..'):
        asset_info[relative_path] = {}
        
        # For each .obj file in this subfolder
        for file in files:
            if file.endswith('.obj'):
                obj_name = os.path.splitext(file)[0]  # Remove .obj extension
                asset_info[relative_path][obj_name] = {
                    'urdf_path': os.path.join(relative_path, obj_name),
                    'density': 1250.0,
                    'friction': 0.5
                }

# Ensure the directory exists
os.makedirs(os.path.dirname(yaml_output_path), exist_ok=True)

# Write to YAML file
with open(yaml_output_path, 'w') as yaml_file:
    yaml.dump(asset_info, yaml_file, default_flow_style=False)

print(f"YAML file generated at {yaml_output_path}")

