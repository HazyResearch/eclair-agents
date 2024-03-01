import json
import os
from tqdm import tqdm

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
                if event['type'] == 'state':
                    # Fix json_state coords
                    json_state = json.loads(event['data'].get('json_state', '[]'))
                    for element in json_state:
                        if 'x' in element and 'y' in element:
                            width_offset = element['width'] / 2
                            height_offset = element['height'] / 2
                            element['x'] -= width_offset
                            element['y'] -= height_offset
                    event['data']['json_state'] = json.dumps(json_state)
                elif event['type'] == 'action':
                    # Fix CLICK coords
                    if 'x' in event['data'] and 'element_attributes' in event['data'] and event['data']['element_attributes'] is not None and 'x' in event['data']['element_attributes']:
                        y_offset = int(event['data']['y'] - event['data']['element_attributes']['y'])
                        x_offset = int(event['data']['x'] - event['data']['element_attributes']['x'])
                        # Update coords
                        event['data']['element_attributes']['x'] += x_offset
                        event['data']['element_attributes']['y'] += y_offset
                        event['data']['element_attributes']['element']['x'] += x_offset
                        event['data']['element_attributes']['element']['y'] += y_offset
                new_trace.append(event)
            json_data['trace'] = new_trace
            # Write JSON
            with open(path_to_json_file, 'w') as f:
                json.dump(json_data, f, indent=2)