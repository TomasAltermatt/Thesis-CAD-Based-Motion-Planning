import os
import argparse

def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f'Removed: {file_path}')
                except Exception as e:
                    print(f'Error removing {file_path}: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recursively remove .DS_Store files from a specified directory.')
    parser.add_argument('--dir', required=True, help='The directory path to clean.')

    args = parser.parse_args()
    directory = args.dir

    remove_ds_store(directory)

