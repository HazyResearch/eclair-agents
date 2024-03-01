import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from eclair.demonstration_collection.postprocess_trace import (
    merge_consecutive_scrolls,
)

path_to_dir: str = '/Users/mwornow/Downloads/test/'

for path_to_task_dir in tqdm(os.listdir(path_to_dir)):
    # check if path_to_task_dir is a directory
    if not os.path.isdir(os.path.join(path_to_dir, path_to_task_dir)):
        continue
    # Loop thru all files in directory
    for path_to_file in os.listdir(os.path.join(path_to_dir, path_to_task_dir)):
        # Find file that ends in '.json' and does not start with '[raw]'
        if path_to_file.endswith('.json') and not path_to_file.startswith('[raw]'):
            path_to_json_file: str = os.path.join(path_to_dir, path_to_task_dir, path_to_file)
            # Load JSON
            with open(path_to_json_file, 'r') as f:
                json_data = json.load(f)
            trace = json_data['trace']
            new_trace = merge_consecutive_scrolls(trace, pixel_margin_of_error=10)
            json_data['trace'] = new_trace
            # Write JSON
            path_to_gt_file: str = os.path.join(os.path.dirname(path_to_json_file), f"{os.path.basename(path_to_json_file)}")
            if os.path.exists(path_to_gt_file):
                print(f"Skipping {path_to_gt_file} because it already exists")
            else:
                with open(path_to_gt_file, 'w') as f:
                    json.dump(json_data, f, indent=2)