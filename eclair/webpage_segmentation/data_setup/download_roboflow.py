"""
Downloads the datasets from Roboflow and prepares them for training.
"""
import os
import json
from tqdm import tqdm
from typing import Any, List, Tuple
from roboflow import Roboflow
import shutil
import cv2

# Paths
BASE_DIR: str = './datasets'
COCO_BASE_NAME: str = 'webpages-coco'
YOLO_BASE_NAME: str = 'webpages-yolov8'
# API Key
ROBOFLOW_API_KEY: str = "82ZUdIOMDqaJ6cW8VLBG"

path_to_coco_dir: str = os.path.join(BASE_DIR, COCO_BASE_NAME)
path_to_coco_single_label_dir: str = os.path.join(BASE_DIR, f'{COCO_BASE_NAME}-single-label')
path_to_yolov8_dir: str = os.path.join(BASE_DIR, YOLO_BASE_NAME)
path_to_yolov8_single_label_dir: str = os.path.join(BASE_DIR, f'{YOLO_BASE_NAME}-single-label')

################################################################
################################################################
#
# COCO
#
################################################################
################################################################

class COCO:
    @staticmethod
    def remap_categories_to_single_label(path_to_dataset_dir: str, new_label: str):
        for split in ['train', 'valid', 'test']:
            if not os.path.exists(os.path.join(path_to_dataset_dir, split)):
                continue
            for filename in os.listdir(os.path.join(path_to_dataset_dir, split)):
                if not filename.endswith('_annotations.coco.json'):
                    continue
                path_to_coco_json: str = os.path.join(path_to_dataset_dir, split, filename)
                with open(path_to_coco_json, 'r') as file:
                    data = json.load(file)
                
                # Remap categories to a single category
                data['categories'] = [ { 'id' : 0, 'name' : new_label, 'supercategory' : 'none' } ] 
                
                # Remap annotations to new category
                for idx in range(len(data['annotations'])):
                    data['annotations'][idx]['category_id'] = 0
                
                # Save updated annotations
                with open(path_to_coco_json, 'w') as file:
                    json.dump(data, file)

    @staticmethod
    def remove_long_filenames(path_to_dataset_dir: str, max_filename_chars: int = 80):
        """Delete all files with filenames that are too long, and adjust corresponding entries in annotations.json """
        for split in ['train', 'valid', 'test']:
            path_to_split_dir: str = os.path.join(path_to_dataset_dir, split)
            if not os.path.exists(path_to_split_dir):
                continue
            for filename in os.listdir(path_to_split_dir):
                if not filename.endswith('_annotations.coco.json'):
                    continue
                path_to_coco_json: str = os.path.join(path_to_dataset_dir, split, filename)
                with open(path_to_coco_json, 'r') as file:
                    data = json.load(file)
                
                # Remove images with long filenames from annotations
                images: List[Any] = []
                for idx in tqdm(range(len(data['images'])), desc=f'Ignoring too long images: `{split}`'):
                    old_filename: str = data['images'][idx]['file_name']
                    if len(old_filename) <= max_filename_chars:
                        images.append(data['images'][idx])
                    else:
                        print(f"Ignoring too long filename: `{old_filename}`")
                data['images'] = images

                # Save updated annotations
                with open(path_to_coco_json, 'w') as file:
                    json.dump(data, file)

    @staticmethod
    def align_image_dims(path_to_dataset_dir: str):
        """Make sure that all images in the dataset have the same dimensions as specified in annotations.json"""
        for split in ['train', 'valid', 'test']:
            path_to_split_dir: str = os.path.join(path_to_dataset_dir, split)
            if not os.path.exists(path_to_split_dir):
                continue
            for filename in os.listdir(path_to_split_dir):
                if not filename.endswith('_annotations.coco.json'):
                    continue
                path_to_coco_json: str = os.path.join(path_to_dataset_dir, split, filename)
                with open(path_to_coco_json, 'r') as file:
                    data = json.load(file)

                # Set image dims in annotations.json to actual image dims
                for idx in tqdm(range(len(data['images'])), desc=f'Aligning image dimensions: `{split}`'):
                    path_to_image: str = os.path.join(path_to_dataset_dir, split, data['images'][idx]['file_name'])
                    image = cv2.imread(path_to_image)
                    if image is None:
                        continue
                    height, width = image.shape[:2]
                    data['images'][idx]['height'] = height
                    data['images'][idx]['width'] = width

                # Save updated annotations
                with open(path_to_coco_json, 'w') as file:
                    json.dump(data, file)

################################################################
################################################################
#
# YOLOv8
#
################################################################
################################################################

class YOLOv8:
    @staticmethod
    def modify_labels(directory):
        # Iterate over all files in the given directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Check if it's a file
            if os.path.isfile(filepath):
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

if __name__ == '__main__':
    # Download ROBOFLOW custom dataset
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("workflowaugmentation").project("webpages-abgy4")
    ## COCO version
    dataset = project.version(7).download(model_format="coco", location=path_to_coco_dir, overwrite=True)
    ## YOLOv8 version
    # dataset = project.version(7).download(model_format="yolov8", location=path_to_yolov8_dir, overwrite=True)

    #
    # COCO setup
    #

    # Remove all images with filenames that are too long from annotations.json
    max_filename_chars: int = 255
    COCO.remove_long_filenames(path_to_coco_dir, max_filename_chars)
    
    # Make sure all image sizes align with annotations.json
    COCO.align_image_dims(path_to_coco_dir)
    

    # Create single label version (i.e. remap all labels to 'webelement')
    shutil.rmtree(path_to_coco_single_label_dir, ignore_errors=True)
    shutil.copytree(path_to_coco_dir, path_to_coco_single_label_dir)
    COCO.remap_categories_to_single_label(path_to_coco_single_label_dir, 'webelement')
    
    #
    # YOLOv8 setup
    #

    # Create single label version (i.e. remap all labels to 'webelement')
    # shutil.rmtree(path_to_yolov8_single_label_dir, ignore_errors=True)
    # shutil.copytree(path_to_yolov8_dir, path_to_yolov8_single_label_dir)
    # COCO.remap_categories_to_single_label(path_to_yolov8_single_label_dir, 'webelement')