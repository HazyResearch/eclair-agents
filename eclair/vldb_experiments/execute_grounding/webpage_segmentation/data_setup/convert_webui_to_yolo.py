"""
This script formats WebUI data into YOLO format.

Example usage: python3 convert_webui_to_yolo.py --version 7k

"""
import os 
import re
import shutil
import argparse
import json
import gzip
from tqdm import tqdm
import PIL
from PIL import Image
import yaml

def get_all_folders_in_directory(directory_path):
    # List all entries in the directory
    all_entries = os.listdir(directory_path)

    # Filter out only the folders
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(directory_path, entry))]

    return folders

def format_webui(version):

    original_webui_dir = os.path.join(os.getenv("HF_HOME"), f"hub/webui-{version}")
    new_webui_dir = os.path.join(os.getenv("HF_HOME"), f"hub/webui-{version}-yolo")

    split_folders = get_all_folders_in_directory(original_webui_dir)

    for split_folder_name in split_folders:

        original_split_dir = os.path.join(original_webui_dir, split_folder_name)

        if "train" in split_folder_name:
            split_name = "train"
        elif "test" in split_folder_name:
            split_name = "test"
        elif "val" in split_folder_name:
            split_name = "valid"

        print(f"### Creating and formatting {split_name} folder: ###")

        # Create new folders for our formatted data to live in        
        split_dir = os.path.join(new_webui_dir, split_name)
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Regular expression to match the required files
        pattern = r"default_(\d+-\d+)-(screenshot-full\.webp|bb\.json\.gz)"

        # Iterate through all folders in data_folder to copy files into the new folder
        for folder in tqdm(os.listdir(original_split_dir), desc="Processing folders"):
            folder_path = os.path.join(original_split_dir, folder)

            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                
                # Create a dictionary to track matching files
                matched_files = {}

                # Find matching pairs of files
                for file in files:
                    match = re.match(pattern, file)
                    if match:
                        prefix, filetype = match.groups()
                        if prefix not in matched_files:
                            matched_files[prefix] = {'webp': None, 'json.gz': None}
                        if filetype == 'screenshot-full.webp':
                            matched_files[prefix]['webp'] = file
                        elif filetype == 'bb.json.gz':
                            matched_files[prefix]['json.gz'] = file

                # Copy matched files to the respective directories
                for prefix, file_types in matched_files.items():
                    if file_types['webp'] and file_types['json.gz']:
                        src_webp = os.path.join(folder_path, file_types['webp'])
                        dest_webp = os.path.join(images_dir, f"{folder}_{file_types['webp']}")
                        shutil.copy(src_webp, dest_webp)

                        src_json = os.path.join(folder_path, file_types['json.gz'])
                        dest_json = os.path.join(labels_dir, f"{folder}_{file_types['json.gz']}")
                        shutil.copy(src_json, dest_json)

        # Iterate through all .json.gz files in the labels directory and unzip them
        for filename in tqdm(os.listdir(labels_dir), desc="Unzipping JSON files"):
            if filename.endswith(".json.gz"):
                file_path = os.path.join(labels_dir, filename)
                output_path = file_path[:-3]  # Remove '.gz' from the filename for the output

                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    
                os.remove(file_path)

        def convert_webp_to_jpg(images_dir, labels_dir):
            for image_path in tqdm(os.listdir(images_dir), desc="Converting images from .webp to .jpg format"):
                try:
                    if image_path.endswith(".webp"):
                        webp_path = os.path.join(images_dir, image_path)
                        jpg_path = os.path.join(images_dir, image_path.replace(".webp", ".jpg"))

                        with Image.open(webp_path) as img:
                            img = img.convert('RGB')
                            img.save(jpg_path, 'jpeg')
                except PIL.UnidentifiedImageError:
                    # If the image is corrupted, remove it and it's corresponding label
                    os.remove(webp_path)
                    label_filename = image_path.replace('screenshot-full.webp', 'bb.json')
                    label_path = f"{labels_dir}/{label_filename}"
                    if os.path.exists(label_path):
                        os.remove(label_path)

        # Convert all .webp images in the directory to .jpg
        convert_webp_to_jpg(images_dir, labels_dir)

        def get_image_dimensions(image_path):
            with Image.open(image_path) as img:
                return img.size


        def convert_to_yolo(json_file_path, output_txt_path, image_path):
            try:
                img_width, img_height = get_image_dimensions(image_path)

                with open(json_file_path, 'r') as file:
                    json_data = json.load(file)

                json_data = {k: v for k, v in json_data.items() if v is not None and v['width'] > 0 and v['height'] > 0}

                yolo_data = []
                for key, bbox in json_data.items():
                    x = bbox['x']
                    y = bbox['y']
                    width = bbox['width']
                    height = bbox['height']

                    center_x = (x + width / 2) / img_width
                    center_y = (y + height / 2) / img_height
                    norm_width = width / img_width
                    norm_height = height / img_height

                    yolo_data.append(f"{key} {center_x} {center_y} {norm_width} {norm_height}")

                with open(output_txt_path, 'w') as file:
                    for line in yolo_data:
                        file.write(line + "\n")

            except PIL.UnidentifiedImageError:
                os.remove(image_path)
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)

        # Convert all JSON files in the directory
        for filename in tqdm(os.listdir(labels_dir), desc="Converting to YOLO format"):
            if filename.endswith(".json"):
                base_name = filename.split('-bb.json')[0]
                json_file_path = os.path.join(labels_dir, filename)
                output_txt_path = os.path.join(labels_dir, base_name + "-bb.txt")
                image_path = os.path.join(images_dir, base_name + "-screenshot-full.webp")
                convert_to_yolo(json_file_path, output_txt_path, image_path)

        def remove_txt_files(directory):
            for filename in os.listdir(directory):
                if filename.endswith(".webp"):
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)


        # Remove all .txt files in the directory
        remove_txt_files(images_dir)

        def rename_label_files(directory):
            for filename in tqdm(os.listdir(directory), desc="Renaming label files"):
                if filename.endswith("-bb.txt"):
                    old_file_path = os.path.join(directory, filename)
                    new_file_path = os.path.join(directory, filename.replace("-bb.txt", ".txt"))
                    
                    os.rename(old_file_path, new_file_path)

        # Rename the files
        rename_label_files(labels_dir)


        def rename_image_files(directory):
            for filename in tqdm(os.listdir(directory), desc="Renaming image files"):
                if filename.endswith("-screenshot-full.jpg"):
                    old_file_path = os.path.join(directory, filename) #-screenshot-full.jpg
                    new_file_path = os.path.join(directory, filename.replace("-screenshot-full.jpg", ".jpg"))
                    
                    os.rename(old_file_path, new_file_path)
        
        # Rename the files
        rename_image_files(images_dir)
        
        def modify_labels(directory):
            # Iterate over all files in the given directory
            for filename in tqdm(os.listdir(directory), desc="Changing labels to 0 for binary classification."):
                filepath = os.path.join(directory, filename)
                
                # Check if it's a file
                if os.path.isfile(filepath) and filename.endswith('.txt'):
                    with open(filepath, 'r') as file:
                        lines = file.readlines()
                    
                    # Modify each line
                    modified_lines = []
                    for line in lines:
                        parts = line.split()
                        if parts:
                            parts[0] = '0'  # Change the first number to 1
                            modified_line = ' '.join(parts)
                            modified_lines.append(modified_line)
                    
                    # Write the modified lines back to the file
                    with open(filepath, 'w') as file:
                        file.write('\n'.join(modified_lines))
        
        modify_labels(labels_dir)

    # Define the YAML content as a dictionary
    yaml_data = {
        'names': ['webelement'],
        'nc': 1,
        'test': '../test/images',
        'train': '../train/images',
        'val': '../valid/images'
    }

    # Specify the directory where you want to save the file
    file_path = os.path.join(new_webui_dir, 'data.yaml')

    # Write the YAML data to the file
    with open(file_path, 'w') as file:
        yaml.dump(yaml_data, file, sort_keys=False)

    print(f"YAML file has been written to {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Format WebUI data into YOLO format.')
    parser.add_argument('--version', type=str, help='Directory of originally downloaded WebUI data',choices=["7k","70k","350k"])

    args = parser.parse_args()

    format_webui(args.version)
    
if __name__ == "__main__":
    main()