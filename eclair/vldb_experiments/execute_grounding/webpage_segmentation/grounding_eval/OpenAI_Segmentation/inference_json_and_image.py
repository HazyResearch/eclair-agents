import pandas as pd
import os
import re 
from openai import OpenAI
from PIL import Image
from eclair.vldb_experiments.execute_grounding.webpage_segmentation.utils import is_center_inside
import ast
import base64
import requests
from eclair.vldb_experiments.execute_grounding.webpage_segmentation.utils import is_center_inside, is_overlapping, calculate_iou
import datetime as datetime
import shutil
import json
from PIL import Image

def convert_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box to YOLO format.

    Parameters:
    - bbox: A dictionary with keys 'x', 'y', 'width', 'height' for the bounding box.
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - A tuple representing the bounding box in YOLO format (x_center, y_center, width, height).
    """
    x_center = (bbox['x'] + bbox['width'] / 2) / img_width
    y_center = (bbox['y'] + bbox['height'] / 2) / img_height
    width = bbox['width'] / img_width
    height = bbox['height'] / img_height

    return (x_center, y_center, width, height)

def run_inference(dataset_dir):
  client = OpenAI()

  df = pd.read_csv(dataset_dir)

  total_correct_predictions = 0
  total_overlapping_predictions = 0
  total_iou = 0

  pred_bbs = []
  responses = []
  centers_within_gt_bb = []
  overlapping_with_gt_bb = []
  ious = []
  prompts = []
  extracted_labels = []

#   images_dir = f"{label_dir}/images"
#   bb_dir = f"{label_dir}/bbs"


  # df = df.head(2)
  for i, row in df.iterrows():
    # Get original image, bounding box, and label description
    image_path = os.path.expanduser(f'~/eclair-agents/eclair/webpage_segmentation/datasets{row["image_path"]}')
    bb = ast.literal_eval(row["bb"])
    description_label = row["description_label"]

    folder_id = os.path.basename(row["image_path"]).split("_")[0]
    image_dim = os.path.basename(row["image_path"])[len(folder_id)+1:].replace(".jpg","")

    axtree_folder = "~/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/WebUI/axtrees"
    bb_ids_folder = "~/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/WebUI/bb_ids"
    axtree_path = os.path.expanduser(os.path.join(axtree_folder,f"{folder_id}_{image_dim}-axtree.json"))
    all_bbs_path = os.path.expanduser(os.path.join(bb_ids_folder,f"{folder_id}_{image_dim}-bb.json"))

    # Read the axtree JSON
    with open(axtree_path, 'r') as f:
        axtree_content = json.load(f)

    # Read the all_bbs JSON
    with open(all_bbs_path, 'r') as f:
        all_bbs = json.load(f)
    
    prompt = f"Given the above image of a webpage, and it's associated accessibility tree below, please output the backendDOMNodeId from the accessibility tree corresponding with the UI element that corresponds with {description_label} on the page. Please put this backendDOMNodeId in quotes. \n\nHere is the accessibility tree: \n{axtree_content}"

    # Function to encode the image
    def encode_image(image_path):
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    api_key = os.environ.get("OPENAI_API_KEY")

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }

    response_json = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    max_attempts = 7
    attempts = 0
    while attempts < max_attempts:
      try:
        print(f"attempt {attempts}")
        response = response_json.json()["choices"][0]["message"]["content"]
        break
      except Exception as e:
        print(f"Exception: {e}, Response: {response_json.json()}")
        attempts += 1
         
    
    print(f"Response: {response}")

    pattern = r"\"(\d+[.,]?\d*)\"" #TODO: may want to accomodate for single quotes too

    matches = re.findall(pattern, response)

    # Extract the last number if any are found
    overlapping = False
    iou = 0
    if matches:
      label_num = matches[-1]
      label_num = re.sub(r'[^\d]', '', label_num)
      print(f"Extracted label: {label_num}")
    
      # Check to see if the predicted bounding box is valid
      if label_num in all_bbs.keys():
        pred_bb_unscaled = all_bbs[label_num]

        # Open the image file
        with Image.open(image_path) as img:
            # Get image dimensions
            img_width, img_height = img.size

        # Convert to yolo format (note: sometimes the block is null, so we have a try except to catch this case)
        try:
          pred_bb = convert_to_yolo(pred_bb_unscaled, img_width, img_height)
          center_within_gt_bb = is_center_inside(bb, pred_bb)
          overlapping = is_overlapping(bb, pred_bb)
          iou = calculate_iou(bb, pred_bb)

          print(f"Predicted_BB {pred_bb}")
          print(f"Is in GT BB? {center_within_gt_bb}")
          print(f"Does Pred BB overlap with GT BB? {overlapping}")
          print(f"IOU: {iou}")
        except Exception as e:
          print(f"e {e}")
          pred_bb = None
          center_within_gt_bb = False
          iou = 0 
        
      else:
        pred_bb = None
        center_within_gt_bb = False
        iou = 0 

      if center_within_gt_bb:
        total_correct_predictions += 1
      
      if overlapping:
        total_overlapping_predictions += 1

      total_iou += iou

    else:
        label_num = None
        pred_bb = None
        center_within_gt_bb = False
        print("No number found in quotes.")

    print(f"Correct BB? {center_within_gt_bb}")

    extracted_labels.append(label_num)
    pred_bbs.append(pred_bb)
    responses.append(response)
    prompts.append(prompt)
    centers_within_gt_bb.append(center_within_gt_bb)
    overlapping_with_gt_bb.append(overlapping)
    ious.append(iou)

  # Add new columns to DataFrame
  df['predicted_bb_yolo'] = pred_bbs
  df["extracted_label"] = extracted_labels
  df['prompt'] = prompts
  df['response'] = responses
  df['center_within_gt_bb'] = centers_within_gt_bb
  df['overlapping_with_gt_bb'] = overlapping_with_gt_bb
  df['iou'] = ious
      
  center_point_in_gt_box_accuracy = total_correct_predictions/len(df)*100
  pred_and_gt_overlap_accuracy = total_overlapping_predictions/len(df)*100
  average_iou = total_iou/len(df)

  print(f"Center point in GT box    | Accuracy: {center_point_in_gt_box_accuracy} %")
  print(f"Pred box & GT box overlap | Accuracy: {pred_and_gt_overlap_accuracy} %")
  print(f"Average IOU: {average_iou}")

  # Format the datetime string
  now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  folder_name = f"json_image_predictions_{os.path.basename(dataset_dir).replace('.csv','')}_{now}"
  csv_name = folder_name+".csv"
  path_to_output_dir = os.path.expanduser(f"~/eclair-agents/eclair/webpage_segmentation/grounding_eval/OpenAI_Segmentation/predictions/")
  
  # Create the output directory if it doesn't exist
  full_output_dir = os.path.join(os.path.expanduser(path_to_output_dir), folder_name)
  os.makedirs(full_output_dir, exist_ok=True)
  
  # Save a copy of this script
  script_path = os.path.realpath(__file__)
  shutil.copy(script_path, full_output_dir)
  
  # Save the inference variables to a CSV
  vars_csv_path = os.path.join(full_output_dir, "inference_variables.csv")
  with open(vars_csv_path, 'w') as f:
      f.write("Variable,Value\n")
      f.write(f"Dataset Dir,{dataset_dir}\n")

  # Save the DataFrame to a new CSV file
  output_csv_path: str = os.path.join(os.path.expanduser(full_output_dir), csv_name)
  df.to_csv(output_csv_path, index=False)

  center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path

  print(f"DataFrame saved to: `{output_csv_path}`")

  return center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path

if __name__ == "__main__":
    
    dataset_dir = "~/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/WebUI/bb_labels_even_split_cropped.csv"

    run_inference(dataset_dir)