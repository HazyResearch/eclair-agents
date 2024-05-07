import argparse
from datetime import datetime
import csv
import os

from eclair.vldb_experiments.execute_grounding.grounding_eval.OpenAI_Segmentation.create_labeled_images import run_create_labeled_images
from eclair.vldb_experiments.execute_grounding.grounding_eval.OpenAI_Segmentation.inference import run_inference as run_gpt4v_set_of_marks_inference
from eclair.vldb_experiments.execute_grounding.grounding_eval.OpenAI_Segmentation.inference_json_and_image import run_inference as run_gpt4v_json_image_inference
from eclair.vldb_experiments.execute_grounding.grounding_eval.OpenAI_Segmentation.inference_json_alone import run_inference as run_gpt4v_json_only_inference
from eclair.vldb_experiments.execute_grounding.grounding_eval.CogAgent.inference import run_inference as run_cogagent_inference
from eclair.utils.helpers import get_rel_path

CONFIDENCE_THRESHOLD = 0.3
CROPPED = True

# Datasets
MIND2WEB = "Mind2Web"
WEBUI = "WebUI"

# Prompting styles
SET_OF_MARKS = "set-of-marks"
AXTREE = "axtree"

# Bounding box generators
GROUND_TRUTH_BB_GENERATOR = "ground_truth"
YOLO_NAS_BB_GENERATOR = "YOLONAS"
NONE = "none"

YOLO_NAS_BB_GENERATOR_MODEL_NAME = "yolo_nas_l_webui"

# Inference model
COGAGENT = "CogAgent"
GPT4V = "GPT4V"

# Results output dir
RESULTS_OUTPUT_FILE_PATH = "results/results.csv"
BASE_DATA_DIR = get_rel_path(__file__, "../../../data/webpage_segmentation_data/")

## WebUI data paths ##
WEBUI_CSV_DIR = os.path.join(BASE_DATA_DIR, "bounding-box-labels/WebUI/bb_labels_even_split_cropped.csv")
WEBUI_BB_DIR = os.path.join(BASE_DATA_DIR, "bounding-box-labels/WebUI/cropped_bbs_even_split")

## Mind2Web data paths ##
MIND2WEB_CSV_DIR = os.path.join(BASE_DATA_DIR, "bounding-box-labels/Mind2Web/bb_labels_mind2web_cropped.csv")
MIND2WEB_BB_DIR = os.path.join(BASE_DATA_DIR, "bounding-box-labels/Mind2Web/cropped_bbs")

def get_dataset_and_bb_dir(dataset):
    if dataset==MIND2WEB:
        return MIND2WEB_CSV_DIR, MIND2WEB_BB_DIR
    elif dataset==WEBUI:
        return WEBUI_CSV_DIR, WEBUI_BB_DIR
    else:
        raise Exception(f"Must specify either {MIND2WEB} or {WEBUI} as your dataset.")

def save_results_to_csv(results, file_path=RESULTS_OUTPUT_FILE_PATH):
    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file does not exist
        if not file_exists:
            headers = [
                "Inference Model", "Prompt Strategy", "Bounding Box Generator",
                "Dataset", "Center Point in GT Box Accuracy",
                "Pred and GT Overlap Accuracy", "Average IOU", "Output CSV Path", "Date/Time"
            ]
            writer.writerow(headers)
        
        # Write the results
        for result in results:
            writer.writerow(result)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model", type=str, default=GPT4V, choices=[COGAGENT, GPT4V])
    parser.add_argument("--quant", type=int, default=None, choices=[4])
    parser.add_argument("--input_image", type=str, default='True', choices=['True','False'])
    parser.add_argument("--prompt_strat", type=str, default=None)
    parser.add_argument("--bbox_generator", type=str, default=NONE, choices=[NONE, GROUND_TRUTH_BB_GENERATOR, YOLO_NAS_BB_GENERATOR])
    parser.add_argument("--regenerate_boxes", type=str, default=False, choices=['True', 'False'])
    parser.add_argument("--dataset", type=str, default=WEBUI, choices=[MIND2WEB, WEBUI])
    return parser.parse_args()

if __name__ == '__main__':
    # Parse args
    args = parse_args()
    inference_model: str = args.inference_model
    prompt_strat: str = args.prompt_strat
    bbox_generator: str = args.bbox_generator
    dataset: str = args.dataset
    regenerate_boxes = args.regenerate_boxes == 'True'
    input_image = args.input_image == 'True'

    # Get the main dataset with our images
    data_dir, bb_dir = get_dataset_and_bb_dir(dataset)

    # If a bbox_generator is specified
    if bbox_generator!=NONE and regenerate_boxes:
        # Use this to generate bbox data
        print(f"Generating bounding boxes with bounding box generator '{bbox_generator}'.")
        set_of_marks_dir = run_create_labeled_images(data_dir=data_dir, bb_dir=bb_dir, dataset=dataset,bounding_box_generator=bbox_generator,confidence_threshold=CONFIDENCE_THRESHOLD, cropped=CROPPED)
    elif bbox_generator!=NONE and not regenerate_boxes:
        # Get previously generated bboxes from the same model
        set_of_marks_dir = os.path.join(BASE_DATA_DIR, f"grounding_eval/openai_bb_{YOLO_NAS_BB_GENERATOR_MODEL_NAME}_conf_{CONFIDENCE_THRESHOLD}_{dataset}")
        if not os.path.exists(set_of_marks_dir):
            raise Exception(f"The directory {set_of_marks_dir} does not exist. ")
    
    print(f"Using dataset {data_dir} for inference.")

    # Now, run inference using either cogagent or gpt4
    if inference_model == GPT4V and prompt_strat == SET_OF_MARKS:
        print(f"Beginning inference with GPT 4V.")
        center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path = run_gpt4v_set_of_marks_inference(data_dir, set_of_marks_dir)
    elif inference_model == GPT4V and prompt_strat == AXTREE and input_image:
        print(f"Beginning inference with inputs: JSON, Image, Model: GPT 4V.")
        center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path = run_gpt4v_json_image_inference(data_dir)
    elif inference_model == GPT4V and prompt_strat == AXTREE and not input_image:
        print(f"Beginning inference with inputs: JSON, Model: GPT 4V.")
        center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path = run_gpt4v_json_only_inference(data_dir)
    elif inference_model == COGAGENT:
        print(f"Beginning inference with CogAgent.")
        center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path = run_cogagent_inference(data_dir, quant=args.quant)
    else:
        raise Exception(f"Model {inference_model} not supported. Please use {GPT4V} or {COGAGENT}.")
    
    # Save results to csv at RESULTS_OUTPUT_FILE_PATH
    # This should include all of the inputs from this file, the outputs from our model, as well as the date / time
    # Assuming you have the results in a list of tuples or lists
    results = [
        (
            inference_model, prompt_strat, bbox_generator, dataset,
            center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy,
            average_iou, output_csv_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    ]
    save_results_to_csv(results)