import os
import zipfile
import shutil

def unzip_files(source_dir, destination_dir):
    # Mapping of zip files to their target directories
    file_map = {
        'test1': 'test',
        'test2': 'test',
        'train1': 'train',
        'train2': 'train',
    }

    # Create destination subdirectories if they don't exist
    for subfolder in file_map.values():
        os.makedirs(os.path.join(destination_dir, subfolder), exist_ok=True)

    # Iterate over each file in the source directory
    for file_name in os.listdir(source_dir):
        zip_path = os.path.join(source_dir, file_name)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Determine the output directory
            output_dir = os.path.join(destination_dir)
            # Extract all the contents of zip file in the directory
            zip_ref.extractall(output_dir)
            print(f"Unzipped {file_name} to {output_dir}")

    for dir_name in os.listdir(destination_dir):
        if dir_name not in file_map:
            continue
        dir_path = os.path.join(destination_dir, dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            shutil.move(file_path, os.path.join(destination_dir, file_map[dir_name], file_name))
        os.rmdir(dir_path)

# Define the source and destination directories
base_dir = '/home/nibjohen/scratch_emmy_project/dancetrack'
source_directory = os.path.join(base_dir, 'zip')
destination_directory = os.path.join(base_dir, 'data')

# Call the function with the paths
unzip_files(source_directory, destination_directory)
