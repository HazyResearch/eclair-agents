import json
import os
from tqdm import tqdm

from eclair.utils.helpers import convert_trace_action_to_dsl

path_to_dir: str = '/Users/mwornow/Downloads/new'

for path_to_task_dir in tqdm(os.listdir(path_to_dir)):
    # check if path_to_task_dir is a directory
    if not os.path.isdir(os.path.join(path_to_dir, path_to_task_dir)):
        continue
    # Loop thru all files in directory
    for path_to_file in os.listdir(os.path.join(path_to_dir, path_to_task_dir)):
        # Find file that ends in '.json' and does not start with '['
        if path_to_file.endswith('.json') and not path_to_file.startswith('['):
            path_to_json_file: str = os.path.join(path_to_dir, path_to_task_dir, path_to_file)
            # Load JSON
            with open(path_to_json_file, 'r') as f:
                json_data = json.load(f)
            trace = json_data['trace']
            new_trace = []
            current_json_state = None
            for event_idx, event in enumerate(trace):
                if event['type'] == 'state':
                    # add gt labels
                    event['gt_labels'] = {}
                    if event_idx + 1 < len(trace):
                        event['gt_labels']['next_action_suggestion'] = 'TODO - CHANGE ME'
                        event['gt_labels']['sop_instruction'] = 'TODO - CHANGE ME'
                        event['gt_labels']['actuation'] = convert_trace_action_to_dsl(trace[event_idx + 1])['actuation_suggestion']['action']
                        # remove avanika labels
                        if 'gt__next_action_suggestion' in event['data']:
                            event['gt_labels']['next_action_suggestion'] = event['data']['gt__next_action_suggestion']
                            del event['data']['gt__next_action_suggestion']
                        if 'gt__actuation' in event['data']:
                            assert event['gt_labels']['actuation'] == event['data']['gt__actuation'], f"{path_to_file} | event['gt_labels']['actuation']: {event['gt_labels']['actuation']}\nevent['data']['gt__actuation']: {event['data']['gt__actuation']}"
                            del event['data']['gt__actuation']
                        if 'gt__sop_instruction' in event['data']:
                            event['gt_labels']['sop_instruction'] = event['data']['gt__sop_instruction']
                            del event['data']['gt__sop_instruction']
                elif event['type'] == 'action':
                    # add gt labels
                    event['gt_labels'] = {}
                    event['gt_labels']['precondition'] = 'TODO - CHANGE ME'
                    if event['data']['type'] == 'mouseup':
                        event['gt_labels']['action_type'] = 'CLICK'
                    elif event['data']['type'] == 'scroll':
                        event['gt_labels']['action_type'] = 'SCROLL'
                    elif event['data']['type'] in ['keystroke']:
                        event['gt_labels']['action_type'] = 'TYPE'
                    elif event['data']['type'] in ['keypress']:
                        event['gt_labels']['action_type'] = 'PRESS'
                    else:
                        raise ValueError(f"Unknown action type: {event['data']['type']} for file: {path_to_file}\n\n{event}")
                    # remove avanika labels
                    if 'actual_recorded_action' in event['data']:
                        assert event['data']['actual_recorded_action'] == event['gt_labels']['action_type'], f"{path_to_file} | {event['data']['actual_recorded_action']} != {event['gt_labels']['action_type']}"
                        del event['data']['actual_recorded_action']
                    if 'gt__precondition' in event['data']:
                        event['gt_labels']['precondition'] = event['data']['gt__precondition']
                        del event['data']['gt__precondition']
                new_trace.append(event)
            json_data['trace'] = new_trace
            # Write JSON
            path_to_gt_file: str = os.path.join(os.path.dirname(path_to_json_file), f"[gt] {os.path.basename(path_to_json_file)}")
            if os.path.exists(path_to_gt_file):
                print(f"Skipping {path_to_gt_file} because it already exists")
            else:
                with open(path_to_gt_file, 'w') as f:
                    json.dump(json_data, f, indent=2)