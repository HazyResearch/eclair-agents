import json
import os
from typing import Dict
from tqdm import tqdm

from eclair.utils.helpers import get_webarena_task_json

path_to_dir: str = '/Users/mwornow/Downloads/[VLDB] 30 WebArena Tasks'

for path_to_task_dir in tqdm(os.listdir(path_to_dir), leave=False):
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
            task_id: int = path_to_file.split(' @ ')[0].replace('[gt] ', '')
            task: Dict[str, str] = get_webarena_task_json(task_id)
            assert task is not None, f"Error - WebArena task {task_id} not found"
            json_data['webarena'] = {
                **task,
            }
            # Write JSON
            with open(path_to_json_file, 'w') as f:
                json.dump(json_data, f, indent=2)