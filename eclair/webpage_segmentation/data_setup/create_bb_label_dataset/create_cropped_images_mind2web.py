# import cv2
import numpy as np
import random
import pandas as pd
import os
import ast
from PIL import Image
import pickle

random.seed(42)


ORIGINAL_DATA_FILE_PATH = "/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/bb_labels_mind2web.csv"
ORIGINAL_BB_DIR = "/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/bbs"

CROPPED_DATA_FILE_PATH = "/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/bb_labels_mind2web_cropped.csv"
CROPPED_IMAGES_DIR = "/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/cropped_images"
CROPPED_BBS_DIR = "/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/cropped_bbs"

def convert_yolo_to_pixels(bb, img_width, img_height):
    """Convert YOLO format bounding box to pixel coordinates."""
    x_center, y_center, width, height = bb
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2

def adjust_bbox_for_crop(original_bbox, crop_top_left, crop_width, crop_height, img_width, img_height):
    """Adjust the bounding box coordinates for the cropped image and convert to YOLO format."""
    x1, y1, x2, y2 = convert_yolo_to_pixels(original_bbox, img_width, img_height)
    crop_left, crop_top = crop_top_left

    # Adjust coordinates relative to the cropped area
    x1_cropped = x1 - crop_left
    y1_cropped = y1 - crop_top
    x2_cropped = x2 - crop_left
    y2_cropped = y2 - crop_top

    # Normalize coordinates to the cropped image dimensions
    x_center_cropped = (x1_cropped + x2_cropped) / 2 / crop_width
    y_center_cropped = (y1_cropped + y2_cropped) / 2 / crop_height
    width_cropped = (x2_cropped - x1_cropped) / crop_width
    height_cropped = (y2_cropped - y1_cropped) / crop_height

    return [x_center_cropped, y_center_cropped, width_cropped, height_cropped]

def bbox_within_crop(bb, crop_left, crop_top, crop_width, crop_height):
    """Check if the bounding box is within the cropped area."""
    x1, y1, x2, y2 = bb
    return (x1 < crop_width + crop_left) and (x2 > crop_left) and (y1 < crop_height + crop_top) and (y2 > crop_top)

def random_crop(image, bbox, all_bbs, screen_width, screen_height): 
    img_width, img_height = image.size
    x1, y1, x2, y2 = convert_yolo_to_pixels(bbox, img_width, img_height)

    # Ensure the bounding box is smaller than the screen size
    if (x2 - x1) > screen_width or (y2 - y1) > screen_height:
        raise ValueError("Bounding box is larger than the screen dimensions.")

    # Randomly choose the top left corner of the cropped area
    # left = random.randint(max(0, x2 - screen_width), min(x1, img_width - screen_width))
    left = 0 # TODO: double check this assumption
    top = random.randint(max(0, y2 - screen_height), min(y1, img_height - screen_height))

    new_main_bbox = adjust_bbox_for_crop(bbox, (left, top), screen_width, screen_height, img_width, img_height)
    new_all_bbs = []

    right = left + screen_width
    bottom = top + screen_height
    
    for bb in all_bbs:
        if bbox_within_crop(convert_yolo_to_pixels(bbox, img_width, img_height), left, top, screen_width, screen_height):
            new_bb = adjust_bbox_for_crop(bb, (left, top), screen_width, screen_height, img_width, img_height)
            new_all_bbs.append(new_bb)

    # Convert the list to a NumPy array
    new_all_bbs = np.array(new_all_bbs)

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image, new_main_bbox, new_all_bbs

max_pixels = 1124 * 1124

df = pd.read_csv(ORIGINAL_DATA_FILE_PATH)

cropped_image_paths = []
cropped_bbs = []

incorrect_img_count = 0 

rows_to_remove = []

for i,row in df.iterrows():
    print(f"i {i}")

    # breakpoint()

    image_path = row["image_path"]
    # label_path = row["label_path"]

    # Open the file and read each line
    all_bb_path = f'{ORIGINAL_BB_DIR}/{os.path.basename(row["image_path"]).replace(".jpeg",".pickle")}'
    with open(all_bb_path, 'rb') as file:
        all_bbs = ast.literal_eval(pickle.load(file))

    bb = ast.literal_eval(row["bb"])[1:]
 
    image = Image.open(image_path)
    img_width, img_height = image.size
    print(f"Curr: Width {img_width} Height {img_height}")

    screen_ratio = 2234 / 3456
    new_img_width = img_width
    new_img_height = int(img_width * screen_ratio)
    print(f"New: Width {new_img_width} Height {new_img_height}")

    if img_height * img_width > max_pixels and new_img_height < img_height:
        # screen_width, screen_height = 3456, 2234
        print(f"RANDOM CROP")
        cropped_image, new_bbox, new_all_bbs = random_crop(image, bb, all_bbs, new_img_width, new_img_height)
    else:
        print(f"ORIGINAL")
        cropped_image = image
        new_bbox = bb
        new_all_bbs = all_bbs

    # Check if the directory does not exist
    if not os.path.exists(CROPPED_BBS_DIR):
        os.makedirs(CROPPED_BBS_DIR)
    
    with open(f'{CROPPED_BBS_DIR}/{os.path.basename(image_path).replace("jpg","pickle").replace("jpeg","pickle")}','wb') as f:
        pickle.dump(new_all_bbs, f, pickle.HIGHEST_PROTOCOL)

    if not os.path.exists(CROPPED_IMAGES_DIR):
        os.makedirs(CROPPED_IMAGES_DIR)
        
    cropped_image_path = os.path.join(CROPPED_IMAGES_DIR, os.path.basename(image_path))

    # cropped_image = Image.fromarray(cropped_image)
    cropped_image.save(cropped_image_path)
    cropped_image_paths.append(cropped_image_path)
    cropped_bbs.append(new_bbox)

df.drop(rows_to_remove, inplace=True)

print(f"{len(cropped_image_paths)}")
print(f"{len(cropped_bbs)}")

df.rename(columns={"image_path":"original_image_path","bb":"original_bb"},inplace=True)

print(f"incorrect_img_count {incorrect_img_count}")
df["image_path"] = cropped_image_paths
df["bb"] = cropped_bbs

# # Save the DataFrame back to its original location
# df.to_csv(ORIGINAL_DATA_FILE_PATH, index=False)

df.to_csv(CROPPED_DATA_FILE_PATH, index=False)
print(f"New cropped dataset saved to: {CROPPED_DATA_FILE_PATH}")