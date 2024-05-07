"""
USAGE SUGGESTIONS

Quantized version:
python inference_hf.py --model THUDM/cogagent-chat-hf --quant 4

Unquantized version:
python inference_hf.py --model THUDM/cogagent-chat-hf --bf16

"""
import argparse
from typing import List, Optional, Dict, Any
import torch
import re
import pandas as pd
import ast
import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer
from eclair.vldb_experiments.execute_grounding.webpage_segmentation.utils import is_center_inside, is_overlapping, calculate_iou, convert_to_yolo, extract_bounding_box
from eclair.utils.helpers import get_rel_path
from datetime import datetime
import shutil

PATH_TO_DATASETS: str = get_rel_path(__file__, "../../datasets/bounding-box-labels/")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="THUDM/cogagent-chat-hf", help='pretrained model')
    parser.add_argument("--tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer')
    parser.add_argument("--path_to_output_dir", type=str, default="~/eclair-agents/eclair/webpage_segmentation/grounding_eval/CogAgent/predictions", help='path to output dir')
    parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    # parser.add_argument("--is_use_cropped", type=bool, default=True)
    # parser.add_argument("--is_use_even_split", type=bool, default=True)
    return parser.parse_args()

def run_inference(data_dir, model="THUDM/cogagent-chat-hf", tokenizer="lmsys/vicuna-7b-v1.5", path_to_output_dir="~/eclair-agents/eclair/webpage_segmentation/grounding_eval/CogAgent/predictions", bf16=False, quant=None):

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    if bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer)

    if quant:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=quant is not None,
            trust_remote_code=True
        ).to(device).eval()

    print("========Use torch type:{} with device:{}========\n\n".format(torch_type, device))

    df = pd.read_csv(data_dir)

    results: List[Dict[str, str]] = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Running inference..."):
        # Load image
        #TODO: fix this eventually -- this part is hardcoded for the schema of the mind2web vs webui dataset
        if "Mind2Web" in row["image_path"]: # TODO: we also need to fix this to make it runnable on other computers 
            path_to_image: str = os.path.expanduser(row["image_path"]) #TODO: ensure this path works across all datasets
        else:
            path_to_image: str = os.path.expanduser("~/eclair-agents/eclair/webpage_segmentation/datasets" + row["image_path"]) #TODO: ensure this path works across all datasets
        image = Image.open(path_to_image).convert('RGB')
        
        # Create prompt
        description_label: str = row["description_label"].replace("\"","\'")
        query: str = f"Can you advise me on how to \"Click on the {description_label}\"? (with grounding)".replace("the the", "the") # sometimes there are duplicate 'the's
        input_by_model: Dict[str, Any] = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
        inputs: Dict[str, Any] = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
        }
        # TODO: what does this mean?
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]
        gen_kwargs: Dict[str, Any] = { "max_length": 2048, "do_sample": False, }
        
        # Run model
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response: str = response.split("</s>")[0]

        # Format of coordination: The bounding box coordinates in the model's input and output use the format [[x1, y1, x2, y2]], 
        # with the origin at the top left corner, the x-axis to the right, and the y-axis downward. (x1, y1) and (x2, y2) are 
        # the top-left and bottom-right corners, respectively, with values as relative coordinates multiplied by 1000 
        # (prefixed with zeros to three digits).
        predicted_bb: Optional[List[int]] = extract_bounding_box(response) # [x, y, x, y] x 1000

        try:
            gt_bb: List[float] = ast.literal_eval(row["bb"])
            predicted_bb_yolo: List[int] = convert_to_yolo(predicted_bb)
            is_center_within_gt_bb: bool = is_center_inside(gt_bb, predicted_bb_yolo)
            is_overlap: bool = is_overlapping(gt_bb, predicted_bb_yolo)
            iou: float = calculate_iou(gt_bb, predicted_bb_yolo)

            results.append({
                'model_response' : response,
                'predicted_bb' : predicted_bb,
                'predicted_bb_yolo' : predicted_bb_yolo,
                'is_center_within_gt_bb' : is_center_within_gt_bb,
                'is_overlap' : is_overlap,
                'iou' : iou,
            })
        except Exception as e:
            print(f"e: {e}")
            results.append({
                'model_response' : response,
                'predicted_bb' : predicted_bb,
            })
    
    # Add results to existing dataframe
    df_new = pd.DataFrame(results)
    # assert df_new.shape[0] == df.shape[0], f"Error -- df's don't match: new={df_new.shape[0]}, old={df.shape[0]}"
    df = pd.concat([df, df_new], axis=1)
    
    # Calculate overall metrics
    center_point_in_gt_box_accuracy: int = df['is_center_within_gt_bb'].mean()
    pred_and_gt_overlap_accuracy: int = df['is_overlap'].mean()
    average_iou: float = df['iou'].mean()    
    print(f"Center point in GT box    | Accuracy: {center_point_in_gt_box_accuracy}")
    print(f"Pred box & GT box overlap | Accuracy: {pred_and_gt_overlap_accuracy}")
    print(f"Average IOU: {average_iou}")

       # Format the datetime string
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Extract the base name for the dataset and create a folder name
    dataset_name = os.path.basename(data_dir).replace(".csv", "")
    folder_name = f"{dataset_name}_quant_{quant}_{now}"
    
    # Create the output directory if it doesn't exist
    full_output_dir = os.path.join(os.path.expanduser(path_to_output_dir), folder_name)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Save the DataFrame
    output_csv_path = os.path.join(full_output_dir, f"{dataset_name}_quant_{quant}_{now}.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"DataFrame saved to: `{output_csv_path}`")
    
    # Save a copy of this script
    script_path = os.path.realpath(__file__)
    shutil.copy(script_path, full_output_dir)
    
    # Save the inference variables to a CSV
    vars_csv_path = os.path.join(full_output_dir, "inference_variables.csv")
    with open(vars_csv_path, 'w') as f:
        f.write("Variable,Value\n")
        f.write(f"Model,{model}\n")
        f.write(f"Tokenizer,{tokenizer}\n")
        f.write(f"Path to Output Dir,{path_to_output_dir}\n")
        f.write(f"BF16,{bf16}\n")
        f.write(f"Quant,{quant}\n")
        f.write(f"Data Dir,{data_dir}\n")

    # Save the DataFrame to a new CSV file
    # dataset_name = os.path.basename(data_dir).replace(".csv","")
    # output_csv_path: str = os.path.join(os.path.expanduser(path_to_output_dir), f"dataset_{dataset_name}_quant_{quant}_{now}.csv")
    # df.to_csv(output_csv_path, index=False)

    # print(f"DataFrame saved to: `{output_csv_path}`")

    # # Save the DataFrame to a new CSV file
    # dataset_name = os.path.basename(data_dir).replace(".csv","")
    # output_csv_path: str = os.path.join(os.path.expanduser(path_to_output_dir), f"dataset_{dataset_name}_quant_{quant}.csv")
    # df.to_csv(output_csv_path, index=False)

    # print(f"DataFrame saved to: `{output_csv_path}`")

    return center_point_in_gt_box_accuracy, pred_and_gt_overlap_accuracy, average_iou, output_csv_path

if __name__ == '__main__':
    args = parse_args()
    model: str = args.model
    tokenizer: str = args.tokenizer
    path_to_output_dir: str = args.path_to_output_dir
    torch_type = torch.bfloat16 if args.bf16 else torch.float16
    quant = args.quant
    data_dir = args.data_dir

    run_inference(model, tokenizer, path_to_output_dir, torch_type, data_dir, quant)