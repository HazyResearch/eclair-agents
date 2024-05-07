import pandas as pd
import os
import re 
from openai import OpenAI
from eclair.vldb_experiments.execute_grounding.webpage_segmentation.utils import is_center_inside
import ast
import base64
import requests
import pickle
from eclair.vldb_experiments.execute_grounding.webpage_segmentation.utils import is_center_inside, is_overlapping, calculate_iou

import datetime as datetime
import shutil

def run_inference(dataset_dir, set_of_marks_labels_dir):
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

  label_dir = set_of_marks_labels_dir
  images_dir = f"{label_dir}/images"
  bb_dir = f"{label_dir}/bbs"


  # df = df.head(2)
  for i, row in df.iterrows():

    # Get original image, bounding box, and label description
    image_path = os.path.expanduser(row["image_path"])
    bb = ast.literal_eval(row["bb"])
    description_label = row["description_label"]

    # Get the labeled image & corresponding bounding boxes
    labeled_image_path = f"{images_dir}/{os.path.basename(image_path)}"
    bbs_path = f"{bb_dir}/{os.path.basename(image_path).replace('jpg','pickle')}"

    prompt = f"As part of the following task, first describe this image. Then, describe where {description_label} is in this image. Finally, provide the numerical label that corresponds with {description_label} on the page, with the number in quotes."

    # Function to encode the image
    def encode_image(image_path):
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(labeled_image_path)

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

    print(f"Prompt: {prompt}")


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

    pattern = r"\"(\d+[.,]?\d*)\""

    matches = re.findall(pattern, response)

    # Extract the last number if any are found
    overlapping = False
    iou = 0
    if matches:
      label_num = matches[-1]
      label_num = re.sub(r'[^\d]', '', label_num)
      print(f"Extracted label: {label_num}")
      label_num = int(label_num)

      # Open the file in binary read mode
      with open(bbs_path, 'rb') as file:
          # Load data from the file
          data = pickle.load(file)

      # Check to see if the predicted bounding box is inside the gt bb
      print(f"len(data) {len(data)} label_num {label_num}")
      if len(data) > label_num:
        pred_bb = data[label_num]

        center_within_gt_bb = is_center_inside(bb, pred_bb)
        overlapping = is_overlapping(bb, pred_bb)
        iou = calculate_iou(bb, pred_bb)

        print(f"Predicted_BB {pred_bb}")
        print(f"Is in GT BB? {center_within_gt_bb}")
        print(f"Does Pred BB overlap with GT BB? {overlapping}")
        print(f"IOU: {iou}")
        
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

  # Save the DataFrame to a new CSV file
  # output_csv_path = os.path.expanduser(f"~/eclair-agents/eclair/webpage_segmentation/grounding_eval/OpenAI_Segmentation/predictions/predictions_{os.path.basename(dataset_dir)}.csv")
  # output_csv_path = f"predictions/predictions_{os.path.basename(label_dir)}.csv"
  # df.to_csv(output_csv_path, index=False)

  # Format the datetime string
  now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  folder_name = f"predictions_{os.path.basename(dataset_dir).replace('.csv','')}_{now}"
  csv_name = folder_name+".csv"
  path_to_output_dir = os.path.expanduser(f"~/eclair-agents/eclair/webpage_segmentation/grounding_eval/OpenAI_Segmentation/predictions/")

  
  # Extract the base name for the dataset and create a folder name
  # dataset_name = os.path.basename(data_dir).replace(".csv", "")
  # folder_name = f"{dataset_name}_quant_{quant}_{now}"
  
  # Create the output directory if it doesn't exist
  full_output_dir = os.path.join(os.path.expanduser(path_to_output_dir), folder_name)
  os.makedirs(full_output_dir, exist_ok=True)
  
  # # Save the DataFrame
  # output_csv_path = os.path.join(full_output_dir, f"{os.path.basename(dataset_dir).replace(".csv",)}_{now}.csv")
  # df.to_csv(output_csv_path, index=False)
  # print(f"DataFrame saved to: `{output_csv_path}`")
  
  # Save a copy of this script
  script_path = os.path.realpath(__file__)
  shutil.copy(script_path, full_output_dir)
  
  # Save the inference variables to a CSV
  vars_csv_path = os.path.join(full_output_dir, "inference_variables.csv")
  with open(vars_csv_path, 'w') as f:
      f.write("Variable,Value\n")
      f.write(f"Dataset Dir,{dataset_dir}\n")
      f.write(f"Set of marks dir,{set_of_marks_labels_dir}\n")

  # Save the DataFrame to a new CSV file
  # dataset_name = os.path.basename(data_dir).replace(".csv","")
  output_csv_path: str = os.path.join(os.path.expanduser(full_output_dir), csv_name)
  df.to_csv(output_csv_path, index=False)

  center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path

  print(f"DataFrame saved to: `{output_csv_path}`")

  return center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cropped", type=bool, default=True)
    # args = parser.parse_args()
    
    dataset_dir = "~/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/bb_labels_cropped.csv"
    set_of_marks_labels_dir = "~/eclair-agents/eclair/webpage_segmentation/grounding_eval/OpenAI_Segmentation/labels/yolo_nas_l_webui_Conf_0.3_gt_False"

    run_inference(dataset_dir, set_of_marks_labels_dir)