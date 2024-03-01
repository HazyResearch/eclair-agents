from typing import Dict, Optional
import os
import json
from eclair.utils.helpers import (
    find_gt_json,
    find_sop_txt,
)
from collections import namedtuple

Inputs = namedtuple('Inputs', ['path_to_screenshots', 'path_to_gt_json', 'path_to_sop', 'gt_task_data', 'sop', 'model_kwargs'])
def load_files_for_task(path_to_task_dir: str) -> Inputs:
    path_to_screenshots: str = os.path.join(path_to_task_dir, "screenshots")
    path_to_gt_json: Optional[str] = find_gt_json(path_to_task_dir)
    path_to_sop: Optional[str] = find_sop_txt(path_to_task_dir)
    assert path_to_gt_json is not None, f"Could not find [gt].json file in {path_to_task_dir}"
    assert path_to_sop is not None, f"Could not find SOP.txt file in {path_to_task_dir}"
    
    # Read gt_trace.json
    gt_task_data: Dict[str, str] = json.load(open(path_to_gt_json, 'r'))

    # Read SOP.txt (if applicable)
    sop: Optional[str] = open(path_to_sop, 'r').read()

    # Load model
    model_kwargs = {
        "model": "gpt-4-vision-preview",
        "temperature": 0.0,
    }
    
    return Inputs(path_to_screenshots,
                    path_to_gt_json,
                    path_to_sop,
                    gt_task_data,
                    sop,
                    model_kwargs)