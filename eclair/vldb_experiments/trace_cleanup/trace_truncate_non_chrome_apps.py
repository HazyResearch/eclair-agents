import json
import os
from tqdm import tqdm

# TODO -- remove any non-Chrome apps from start/end of video (but not in them middle); adjust trace, trim video, and resample screenshots

path_to_dir: str = '/Users/mwornow/Downloads/[VLDB] 30 WebArena Tasks'

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
            new_trace = []
            current_json_state = None
            for event in trace:
                # TODO
                new_trace.append(event)
            json_data['trace'] = new_trace
            # Write JSON
            with open(path_to_json_file, 'w') as f:
                json.dump(json_data, f, indent=2)