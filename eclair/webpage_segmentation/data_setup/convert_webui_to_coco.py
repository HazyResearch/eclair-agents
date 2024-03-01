from tqdm import tqdm
import os
import gzip
import json
import datetime
from PIL import Image
from typing import Dict

PATH_TO_OUTPUT_DIR: str = './outputs/'
PATH_TO_WEBUI_DATASET_DIR: str = os.path.join(
    os.getenv("HF_DATASETS_CACHE"),
    "webui-7k",
)

# Create output directories
os.makedirs(os.path.join(PATH_TO_OUTPUT_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(PATH_TO_OUTPUT_DIR, 'test'), exist_ok=True)
os.makedirs(os.path.join(PATH_TO_OUTPUT_DIR, 'valid'), exist_ok=True)

annotations_json: Dict[str, str] = {
    "info": {
        "description": "webui-7k",
        "date_created": datetime.datetime.now().isoformat(),
    },
    "categories": [],
    "images": [],
    "annotations": [],
}

img_id_2_idx: Dict[str, int] = {} # [key] = unique image ID, [value] = index in `annotations_json['images']`
for folder in tqdm(os.listdir(PATH_TO_WEBUI_DATASET_DIR)):
    for file in os.listdir(os.path.join(PATH_TO_WEBUI_DATASET_DIR, folder)):
        # Get type of file: axtree, bb, box, class, html, links, screenshot-full, screenshot, style, url, viewport
        file_type: str = file.split('-')[-1]
        file_type: str = file_type.split('.')[0]
        # Get screen: default, iPhone, iPad
        screen: str = file.split('_')[0]
        # Get screen resolution: 1280x720, 1920x1080, 1024x768
        screen_resolution: str = 'x'.join(file.split('_')[1].split('-')[:2])
        # Get file extension
        file_extension: str = file.split('.')[-1]
        # Generate unique image ID
        img_id: str = {folder}-{screen}-{screen_resolution}
        if img_id not in img_id_2_idx:
            img_id_2_idx[img_id] = len(annotations_json['images'])
        # Read file
        path_to_file: str = os.path.join(PATH_TO_WEBUI_DATASET_DIR, folder, file)
        if file_type == 'axtree':
            # convert accessibility labels -> coco class labels 
            with gzip.open(path_to_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            for element_id, label in data.items():
                annotations_json['categories'].append({ # TODO - make a set to join together same label
                    'id' : len(annotations_json['categories']),
                    'name' : label,
                    "supercategory": "none"
                })
        elif file_type == 'bb':
            # convert bounding box coordinates to coco format
            with gzip.open(path_to_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            for element_id, bb in data.items():
                annotations_json['annotations'].append({
                    'id' : len(annotations_json['annotations']),
                    'image_id' : img_id_2_idx[img_id],
                    'category_id' : None, # TODO
                    'bbox' : [
                        bb['x'],
                        bb['y'],
                        bb['width'],
                        bb['height'],
                    ],
                    'area' : bb['width'] * bb['height'],
                    'segmentation' : [],
                    'iscrowd' : 0,
                })
        elif file_type == 'screenshot':
            # convert .webp -> .png
            img = Image.open(path_to_file)
            img.save(os.path.join(PATH_TO_OUTPUT_DIR, f'{img_id}.png'))
            width, height = img.size
            annotations_json['images'].append({
                "id": img_id_2_idx[img_id],
                "file_name": f'{img_id}.png',
                "height": height,
                "width": width,
            })
# Write out `annotations.coco.json` file
with open(os.path.join(PATH_TO_OUTPUT_DIR, '_annotations.coco.json'), 'w') as f:
    json.dump(annotations_json, f, indent=2)