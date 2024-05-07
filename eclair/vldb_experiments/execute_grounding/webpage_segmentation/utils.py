import re
from typing import List, Optional, Any, Dict

def convert_to_yolo(predicted_bb: List[int]) -> List[int]:
    # Unpacking the predicted bounding box
    x1, y1, x2, y2 = [coord / 1000 for coord in predicted_bb]  # Normalize the coordinates
    
    # Calculating the center, width, and height
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    return [x_center, y_center, width, height]

def extract_bounding_box(response) -> Optional[List[int]]:
    # Regex pattern to find numbers inside double square brackets
    pattern = r"\[\[(\d+),(\d+),(\d+),(\d+)\]\]"
    match = re.search(pattern, response)

    bounding_box: Optional[List[int]] = None
    if match:
        # Extract and convert the bounding box coordinates to integers
        bounding_box = [int(coord) for coord in match.groups()]
    return bounding_box

def unpack_bb(bb):
    if len(bb) >4:
        # Unpacking the ground truth bounding box and predicted YOLO bb
        _, gt_x_center, gt_y_center, gt_width, gt_height = bb
    else:
        gt_x_center, gt_y_center, gt_width, gt_height = bb
    return gt_x_center, gt_y_center, gt_width, gt_height
    

def is_center_inside(gt_bb, predicted_bb) -> bool:
    gt_x_center, gt_y_center, gt_width, gt_height = unpack_bb(gt_bb)
    pred_x_center, pred_y_center, _, _ = unpack_bb(predicted_bb)

    # Check if the center of predicted_bb is within gt_bb
    return (
        gt_x_center - gt_width / 2 <= pred_x_center <= gt_x_center + gt_width / 2 and
        gt_y_center - gt_height / 2 <= pred_y_center <= gt_y_center + gt_height / 2
    )

def is_overlapping(bb1, bb2) -> bool:
    bb1_width, bb1_height = bb1[2], bb1[3]
    bb2_width, bb2_height = bb2[2], bb2[3]

    # Calculate the corners of the first bounding box
    bb1_left = bb1[0] - bb1_width / 2
    bb1_right = bb1[0] + bb1_width / 2
    bb1_top = bb1[1] - bb1_height / 2
    bb1_bottom = bb1[1] + bb1_height / 2

    # Calculate the corners of the second bounding box
    bb2_left = bb2[0] - bb2_width / 2
    bb2_right = bb2[0] + bb2_width / 2
    bb2_top = bb2[1] - bb2_height / 2
    bb2_bottom = bb2[1] + bb2_height / 2

    # Check if the bounding boxes overlap
    return (bb1_right > bb2_left and bb1_left < bb2_right and
            bb1_bottom > bb2_top and bb1_top < bb2_bottom)

def calculate_iou(gt_bb, pred_bb):
    # Unpack the bounding boxes
    gt_x_center, gt_y_center, gt_width, gt_height = unpack_bb(gt_bb)
    pred_x_center, pred_y_center, pred_width, pred_height = unpack_bb(pred_bb)

    # Calculate the coordinates of the intersection rectangle
    inter_left = max(gt_x_center - gt_width / 2, pred_x_center - pred_width / 2)
    inter_right = min(gt_x_center + gt_width / 2, pred_x_center + pred_width / 2)
    inter_top = max(gt_y_center - gt_height / 2, pred_y_center - pred_height / 2)
    inter_bottom = min(gt_y_center + gt_height / 2, pred_y_center + pred_height / 2)

    # Calculate the intersection area
    inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)

    # Calculate the areas of both bounding boxes
    gt_area = gt_width * gt_height
    pred_area = pred_width * pred_height

    # Calculate the union area
    union_area = gt_area + pred_area - inter_area

    # Compute the IOU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def clicks_inside_bb_and_no_others(gt_bb, predicted_bb, all_bb):
    # Unpack bounding boxes
    gt_x_center, gt_y_center, gt_width, gt_height = unpack_bb(gt_bb)
    pred_x_center, pred_y_center, pred_width, pred_height = unpack_bb(predicted_bb)

    # Check overlap with ground truth
    overlaps_gt = is_overlapping((gt_x_center, gt_y_center, gt_width, gt_height), 
                            (pred_x_center, pred_y_center, pred_width, pred_height))
    
    print(f"Overlaps GT: {overlaps_gt}")

    if overlaps_gt:
        # Check overlap with other bounding boxes
        for obb in all_bb:
            if not (obb == gt_bb).all():
                if is_overlapping(obb, predicted_bb):
                    print(f"Predicted bb {predicted_bb} overlaps {obb}")
                    return False

    return overlaps_gt