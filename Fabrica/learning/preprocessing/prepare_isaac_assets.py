import os


def scale_obj_file(input_path, output_path, scale_factor):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # Process vertex coordinates (lines that start with 'v ')
            if line.startswith('v '):
                parts = line.strip().split()
                # Scale x, y, z coordinates
                x, y, z = map(float, parts[1:4])
                if isinstance(scale_factor, list):
                    assert len(scale_factor) == 3, "Scale factor must have 3 values for x, y, and z."
                    x_scale, y_scale, z_scale = map(float, scale_factor)
                    scaled_line = f"v {x * x_scale} {y * y_scale} {z * z_scale}\n"
                else:
                    scaled_line = f"v {x * scale_factor} {y * scale_factor} {z * scale_factor}\n"
                outfile.write(scaled_line)
            else:
                # Copy other lines as-is
                outfile.write(line)


def prepare_isaac_assets(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for subdir_name in os.listdir(input_dir):
        input_subdir = os.path.join(input_dir, subdir_name)
        output_subdir = os.path.join(output_dir, subdir_name)
        if os.path.isdir(input_subdir):
            os.makedirs(output_subdir, exist_ok=True)
            for obj_name in os.listdir(input_subdir):
                if obj_name.endswith('.obj'):
                    obj_path_input = os.path.join(input_subdir, obj_name)
                    obj_path_output = os.path.join(output_subdir, obj_name)
                    scale_obj_file(obj_path_input, obj_path_output, 0.01)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    prepare_isaac_assets(args.input_dir, args.output_dir)

