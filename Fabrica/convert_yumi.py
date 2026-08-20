import os
import trimesh

# Define the paths based on our setup
input_meshes_dir = "assets/yumi/raw_meshes" 
output_dir = "assets/yumi"  

os.makedirs(os.path.join(output_dir, "visual"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "collision"), exist_ok=True)

print("Starting YuMi mesh conversion...")

# 1. Convert Visuals (The root .stl files)
for file in os.listdir(input_meshes_dir):
    if file.endswith(".stl"):
        mesh = trimesh.load(os.path.join(input_meshes_dir, file))
        out_name = file.replace(".stl", ".obj")
        mesh.export(os.path.join(output_dir, "visual", out_name))
        print(f"Converted visual: {out_name}")

# 2. Convert Collisions (The coarse/ .stl files)
coarse_dir = os.path.join(input_meshes_dir, "coarse")
if os.path.exists(coarse_dir):
    for file in os.listdir(coarse_dir):
        if file.endswith(".stl"):
            mesh = trimesh.load(os.path.join(coarse_dir, file))
            out_name = file.replace(".stl", ".obj")
            mesh.export(os.path.join(output_dir, "collision", out_name))
            print(f"Converted collision: {out_name}")
            
print("\nConversion complete! You can now safely delete the 'raw_meshes' folder.")