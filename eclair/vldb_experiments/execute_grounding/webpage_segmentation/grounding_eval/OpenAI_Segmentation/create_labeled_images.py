import os
import yaml
import torch
from super_gradients.training import models
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import IPython.display as display
from PIL import ImageDraw, ImageFont, Image
import math
import pickle
import argparse
from tqdm import tqdm

## Parameters ##
GROUND_TRUTH_BB_GENERATOR = "ground_truth"
YOLO_NAS_BB_GENERATOR = "yolo_nas_l_webui"

MIND2WEB = "Mind2Web"
WEBUI = "WebUI"

def get_model_config_and_checkpoint_path(bounding_box_generator):
    config_path, model_checkpoint_path = '',''
    if bounding_box_generator == YOLO_NAS_BB_GENERATOR:
        config_path =  os.path.expanduser(f'/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/models/YOLONAS/experiment_configs/{bounding_box_generator}.yml')
        model_checkpoint_path = "/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/models/YOLONAS/checkpoints/webui_formatted_inner_box/RUN_20231214_114855_038631/average_model.pth"
    else:
        raise Exception(f"No model config or checkpoint path have been specified for the bounding box generator '{bounding_box_generator}'")
    return config_path, model_checkpoint_path

# Function to extend a point towards another point by a certain distance
def extend_point(start, end, distance):
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return end[0] + distance * math.cos(angle), end[1] + distance * math.sin(angle)

# Assuming bbs is a numpy array with each row in YOLO format [c_x, c_y, w, h]
def convert_yolo_to_xyxy(bbs, image_width, image_height):
    # Convert from relative to absolute coordinates
    bbs[:, [0, 2]] *= image_width  # Convert x and width
    bbs[:, [1, 3]] *= image_height  # Convert y and height

    breakpoint()

    # Calculate XYXY format
    xyxy_bbs = np.zeros_like(bbs)
    xyxy_bbs[:, 0] = bbs[:, 0] - bbs[:, 2] / 2  # x1 = c_x - w/2
    xyxy_bbs[:, 1] = bbs[:, 1] - bbs[:, 3] / 2  # y1 = c_y - h/2
    xyxy_bbs[:, 2] = bbs[:, 0] + bbs[:, 2] / 2  # x2 = c_x + w/2
    xyxy_bbs[:, 3] = bbs[:, 1] + bbs[:, 3] / 2  # y2 = c_y + h/2

    return xyxy_bbs

# Function to draw an arrow
def draw_arrow(draw, start, end):

    end = extend_point(start, end, 4)

    # Line
    draw.line([start, end], fill="black", width=1)
    
    # Arrow head
    arrow_head_size = 3
    angle = math.atan2(start[1] - end[1], start[0] - end[0])  # Angle for arrow head direction
    arrow_points = [
        (end[0] + arrow_head_size * math.cos(angle - math.pi / 6), 
        end[1] + arrow_head_size * math.sin(angle - math.pi / 6)),
        end,
        (end[0] + arrow_head_size * math.cos(angle + math.pi / 6),
        end[1] + arrow_head_size * math.sin(angle + math.pi / 6))
    ]
    draw.polygon(arrow_points, fill="black")

# Function to find the closest edge point of a bounding box from a given point
def closest_edge_point(x, y, xmin, ymin, xmax, ymax):
    closest_x = min(max(xmin, x), xmax)
    closest_y = min(max(ymin, y), ymax)
    return closest_x, closest_y

# Function to determine the edge of the label box closest to the bounding box
def label_edge_point(label_x, label_y, text_width, text_height, position):
    if position == "left":
        return (label_x + text_width, label_y + text_height / 2)
    elif position == "top":
        return (label_x + text_width / 2, label_y + text_height)
    elif position == "right":
        return (label_x, label_y + text_height / 2)
    elif position == "bottom":
        return (label_x + text_width / 2, label_y)

def find_empty_space(xmin, ymin, xmax, ymax, occupied_spaces, text_width, text_height):
    label_positions = [
        ((xmin - text_width - 2, ymin), "left"),  # Left of the bounding box
        ((xmin, ymin - text_height - 2), "top"),  # Above the bounding box
        ((xmax + 2, ymin), "right"),              # Right of the bounding box
        ((xmin, ymax + 2), "bottom")             # Below the bounding box
    ]
    for (label_x, label_y), position in label_positions:
        label_area = (label_x, label_y, label_x + text_width, label_y + text_height)
        if not any(overlap(label_area, box) for box in occupied_spaces):
            return (label_x, label_y), position
    return None, None

def overlap(area1, area2):
    """Check if two areas (x1, y1, x2, y2) overlap."""
    x1, y1, x2, y2 = area1
    x1b, y1b, x2b, y2b = area2
    return not (x2 < x1b or x2b < x1 or y2 < y1b or y2b < y1)

def convert_to_xyxy(json_data, boxes_to_remove):
    # Initialize an empty list for the converted bounding boxes
    converted_bounding_boxes = []

    for idx, bbox in json_data.items():
        if idx not in boxes_to_remove:
            # Extract values from the bbox dictionary
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']

            # Convert back to x1, y1, x2, y2 format
            x1 = x - width / 2
            y1 = y - height / 2
            x2 = x + width / 2
            y2 = y + height / 2

            # Append to the list
            converted_bounding_boxes.append([x1, y1, x2, y2])

    # Convert the list of bounding boxes to a PyTorch tensor
    return np.array(converted_bounding_boxes)

def filter_to_inner_boxes(bbs):
    """
    Input: Expects bb in xyxy format
    Output: Bounding boxes in xyxy format
    """

    # print(f"bbs {bbs}")
    # Convert to x, y, width, height format
    xywh_bounding_boxes = np.zeros_like(bbs)
    # xywh_bounding_boxes = torch.zeros_like(bbs)
    xywh_bounding_boxes[:, 0] = bbs[:, 0] + (bbs[:, 2] - bbs[:, 0]) / 2# x remains the same
    xywh_bounding_boxes[:, 1] = bbs[:, 1] + (bbs[:, 3] - bbs[:, 1]) / 2# y remains the same
    xywh_bounding_boxes[:, 2] = bbs[:, 2] - bbs[:, 0] # width = x2 - x1
    xywh_bounding_boxes[:, 3] = bbs[:, 3] - bbs[:, 1] # height = y2 - y1

    json_data = {}

    # Parse the data into a json 
    for idx, row in enumerate(xywh_bounding_boxes):
        x, y, width, height = row[0],row[1],row[2],row[3]
        bbox = {"x": float(x), "y": float(y), "width": float(width), "height": float(height)}
        json_data[int(idx)] = bbox

    boxes_to_remove = []

    # Assuming bounding_boxes is a list of dictionaries
    for box_a_id, box_a in json_data.items():
        for box_b_id, box_b in json_data.items():
            if box_a_id != box_b_id and box_a_id not in boxes_to_remove and box_b_id not in boxes_to_remove:
                box_a_x_start = round(box_a['x'] - box_a['width'] / 2, 4)
                box_a_x_end = round(box_a['x'] + box_a['width'] / 2, 4)
                box_a_y_start = round(box_a['y'] - box_a['height'] / 2, 4)
                box_a_y_end = round(box_a['y'] + box_a['height'] / 2, 4)

                box_b_x_start = round(box_b['x'] - box_b['width'] / 2, 4)
                box_b_x_end = round(box_b['x'] + box_b['width'] / 2, 4)
                box_b_y_start = round(box_b['y'] - box_b['height'] / 2, 4)
                box_b_y_end = round(box_b['y'] + box_b['height'] / 2, 4)

                # If box B is within box A, remove box A
                if box_a_x_start <= box_b_x_start and box_b_x_end <= box_a_x_end and box_a_y_start <= box_b_y_start and box_b_y_end <= box_a_y_end:
                    boxes_to_remove.append(box_a_id)

    boxes_to_remove = set(boxes_to_remove)
    for box_id in boxes_to_remove:
        json_data.pop(box_id, None)
    filtered_bounding_boxes = convert_to_xyxy(json_data, boxes_to_remove)
    return filtered_bounding_boxes

def run_create_labeled_images(data_dir, bb_dir, dataset, bounding_box_generator, confidence_threshold, cropped):

    if bounding_box_generator != GROUND_TRUTH_BB_GENERATOR:
        # Load model
        CONFIG_FILE_PATH, MODEL_CHECKPOINT_PATH = get_model_config_and_checkpoint_path(bounding_box_generator) 
        # config_file_path = os.path.expanduser(f'/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/models/YOLONAS/experiment_configs/{args.bounding_box_generator}.yml')
        with open(CONFIG_FILE_PATH, 'r') as file:
            config = yaml.safe_load(file)

        model_name = config['model']['name']

        EXPERIMENT_NAME = config['experiment']['name']
        MODEL_ARCH = config['model']['architecture']
        NUM_CLASSES = config['data']['classes']
        DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
        print(f"using device {DEVICE}")

        # Load our trained model
        best_model = models.get(
            MODEL_ARCH,
            num_classes=len(NUM_CLASSES),
            checkpoint_path=MODEL_CHECKPOINT_PATH #TODO: figure out how to get the run location -- just access the folder in the experiment_name dir
        ).to(DEVICE)

    df = pd.read_csv(data_dir)

    # Check if the directory does not exist
    if not os.path.exists("labels"):
        # Create the directory
        os.makedirs("labels")

    # Check if the directory does not exist
    save_dir = os.path.expanduser(f"~/eclair-agents/eclair/webpage_segmentation/grounding_eval/OpenAI_Segmentation/labels/bb_{bounding_box_generator}_{dataset}")
    if bounding_box_generator != GROUND_TRUTH_BB_GENERATOR:
        save_dir = os.path.expanduser(f"~/eclair-agents/eclair/webpage_segmentation/grounding_eval/OpenAI_Segmentation/labels/bb_{bounding_box_generator}_conf_{confidence_threshold}_{dataset}")

    if not os.path.exists(save_dir):
        # Create the directory
        os.makedirs(save_dir)

    # Check if the directory does not exist
    bb_save_dir = f"{save_dir}/bbs"
    if not os.path.exists(bb_save_dir):
        # Create the directory
        os.makedirs(bb_save_dir)

    image_save_dir = f"{save_dir}/images"
    if not os.path.exists(image_save_dir):
        # Create the directory
        os.makedirs(image_save_dir)

    for i, row in tqdm(df.iterrows(), desc="Applying set of marks labels to images."):

        # image_path = os.path.expanduser("~/eclair-agents/eclair/webpage_segmentation/datasets") + row["image_path"]
        # Load image
        #TODO: fix this eventually -- this part is hardcoded for the schema of the mind2web vs webui dataset
        if "Mind2Web" in row["image_path"]: # TODO: we also need to fix this to make it runnable on other computers 
            image_path: str = os.path.expanduser(row["image_path"]) #TODO: ensure this path works across all datasets
        else:
            image_path: str = os.path.expanduser("~/eclair-agents/eclair/webpage_segmentation/datasets" + row["image_path"]) #TODO: ensure this path works across all datasets
        image = Image.open(image_path).convert('RGB')
        print(f"opened image.")

        # image_path = "~/eclair-agents/eclair/webpage_segmentation/datasets" + row["image_path"]
        # bb_path = os.path.expanduser(row["label_path"])

        # Load the image using Pillow
        # image = Image.open(image_path)
        image_width, image_height = image.size

        # Get bounding boxes
        # If we're using a model to generate them, get them by predicting them
        if bounding_box_generator != GROUND_TRUTH_BB_GENERATOR:
            print(f"generating predictions.")
            result = list(best_model.predict(image_path, conf=confidence_threshold))[0]
            bbs = result.prediction.bboxes_xyxy
            print(f"predictions done.")
        # Otherwise, read in our pickled ground truth bounding boxes
        else:
            bbs = []
            # Initialize an empty list to store the bounding boxes
            # if dataset==WEBUI:
                # Load the data back from the pickle file
            print(f"opening bbs.")
            with open(os.path.expanduser(f"{bb_dir}/{os.path.basename(image_path).replace('jpg','pickle')}"), 'rb') as f:
                bbs = pickle.load(f)
            print(f"bbs opened.")
            breakpoint()
            # elif dataset==MIND2WEB:
            #     with open(bb_path, 'rb') as f:
            #         bbs = ast.literal_eval(pickle.load(f))
            # else:
            #     raise Exception(f"Dataset {dataset} not yet supported. Please use {WEBUI} or {MIND2WEB}.")
            
            bbs = np.array(bbs)
            bbs = convert_yolo_to_xyxy(bbs, image_width, image_height)

        # Remove irrelevant outer boxes
        bbs_filtered = filter_to_inner_boxes(bbs) #result.prediction.bboxes_xyxy
        # print(f"bbs_filtered {bbs_filtered}")

        display.display(image)

        # Initialize the drawing context with your image as background
        draw = ImageDraw.Draw(image)

        # Use a specified font
        font_size = 20
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)

        occupied_spaces = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax in bbs_filtered]

        bb_yolo_dict = {}

        # Iterate over bounding box data and draw each box and label
        for i, (xmin, ymin, xmax, ymax) in enumerate(bbs_filtered):
            # Draw the bounding box
            # print(f"{(xmin, ymin, xmax, ymax)}")
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=1)

            # Text to be displayed
            text = f"{i}"

            # Normalize the bounding box coordinates
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            # Update the dictionary with the bounding box in YOLO format
            bb_yolo_dict[i] = [x_center, y_center, width, height]

            # Estimate text width and height
            text_width = len(text) * font_size * 0.6
            text_height = font_size

            label_pos, position = find_empty_space(xmin, ymin, xmax, ymax, occupied_spaces, text_width, text_height)

            if label_pos:
                text_x, text_y = label_pos

                # Update occupied_spaces to include the area of the new label
                label_area = (text_x, text_y, text_x + text_width, text_y + text_height)
                occupied_spaces.append(label_area)

                # Draw a rectangle behind the text for visibility
                draw.rectangle([(text_x - 2, text_y - 2), (text_x + text_width + 2, text_y + text_height + 2)], fill="lightyellow", outline="black")

                # Draw the text
                draw.text((text_x, text_y), text, fill="red", font=font)

                # Find the closest point on the bounding box edge from the label
                closest_point = closest_edge_point(text_x + text_width / 2, text_y + text_height / 2, xmin, ymin, xmax, ymax)

                # Determine the starting point of the arrow from the label box edge
                arrow_start = label_edge_point(text_x, text_y, text_width, text_height, position)

                # Draw an arrow from the label edge to the closest edge point
                draw_arrow(draw, arrow_start, closest_point)
            
        labeled_image_path = f"{image_save_dir}/{os.path.basename(image_path)}"
        # print(f"labeled_images {labeled_image_path}")

        image.save(labeled_image_path)

        bb_path = f"{bb_save_dir}/{os.path.basename(image_path).replace('jpg','pickle')}"
        with open(bb_path, 'wb') as handle:
            pickle.dump(bb_yolo_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return save_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounding_box_generator", type=str, default=YOLO_NAS_BB_GENERATOR, choices=[GROUND_TRUTH_BB_GENERATOR, YOLO_NAS_BB_GENERATOR])
    parser.add_argument("--cropped", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default=WEBUI, choices=[MIND2WEB, WEBUI])
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    args = parser.parse_args()

    run_create_labeled_images(args.dataset, args.bounding_box_generator, args.confidence_threshold, args.cropped)