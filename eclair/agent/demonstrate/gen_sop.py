"""
Usage:

python gen_sop.py './data/Nursing/sitter @ 2024-03-21-10-08-24'
"""
import base64
from io import BytesIO
import os
import shutil

from tqdm import tqdm
from eclair.utils.helpers import (
    _fetch_openai_completion,
    convert_trace_action_to_dsl,
    encode_image,
    get_path_to_screenshots_dir,
    get_path_to_trace_json,
)
from PIL import Image
from typing import Dict, List, Optional, Any
import json
import argparse
from eclair.hospital_demo.demonstrate.prompts import (
    prompt__td_kf_act_intro,
    prompt__td_kf_act_close,
    prompt__td_kf_act_intro__pairwise,
    prompt__td_kf_act_close__pairwise,
    prompt__td_kf_act_intro__pairwise__cropped,
    prompt__join_pairwise,
    prompt__generalize
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_demo_folder", type=str)
    parser.add_argument("--task_descrip", type=str, default=None, help="Short English description of workflow. If not provided, default to value in trcae['task']['description']" )
    parser.add_argument("--is_act", action="store_true", default=False, help="If TRUE, then include action trace" )
    parser.add_argument("--is_pairwise", action="store_true", default=False, help="If TRUE, then instead of prompting all screenshots / action traces at once, prompt two at a time (before/after), and piece together SOP afterwards" )
    parser.add_argument("--is_crop_to_action", action="store_true", help="If TRUE, then crop the screenshot to where the action's coordinates are located.")
    parser.add_argument("--is_verbose", action="store_true", help="If TRUE, then print out stuff")
    return parser.parse_args()

def generate_sop_from_demo(path_to_demo_folder: str, 
                            task_descrip: Optional[str], 
                            is_act: bool,
                            is_pairwise: bool,
                            is_crop_to_action: bool) -> str:
    """Generates an SOP from a video recording

    Args:
        path_to_demo_folder (str): Path to the folder containing the demo. Should contains a `trace.json` and a `screenshots` directory.
        task_descrip (str): English description of task (e.g. "Sitter order")
        is_pairwise (bool): If True, uses a pairwise prompting strat -- e.g. (S, A, S') instead of (S, A, S', A', S'', A''...)
        is_act (bool): If True, then include action trace (e.g. "Click on 'Order' button"
        is_crop_to_action (bool): If True, then crop the screenshot to where the action's coordinates are located.
    """
    ui_name = "Epic EHR" # TODO - pull from trace.json
    crop_box_width: int = 300 # crop box width (px) of screenshot around action coordinates -- i.e. creates a 100x100px box around action coordinates

    # Create output directory
    demo_name: str = os.path.basename(path_to_demo_folder.strip('/').split("/")[-1])
    path_to_output_dir: str = os.path.join(path_to_demo_folder, "sop")
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Get .json trace
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    full_trace: Dict[str, str] = json.loads(open(path_to_trace, "r").read())
    trace_json: List[Dict[str, str]] = full_trace['trace']
    
    # Get task description from trace.json (if not manually specified as arg)
    if task_descrip is None:
        path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
        task_descrip: str = full_trace.get('task', {}).get('description', None)

    # Get screenshots
    path_to_screenshots_dir: str = get_path_to_screenshots_dir(path_to_demo_folder)
    screenshots: List[str] = os.listdir(path_to_screenshots_dir)
    screenshots = sorted(screenshots) # sort by filename, from 0 -> N

    # Loop through trace, interleaving screenshots (states) and actions
    prompt_s_a_sequence: List[str] = []
    cropped_images_base64: List[str] = [] # If is_crop_to_action=True, then store croped images
    for item_idx, item in enumerate(trace_json):
        if item['type'] == 'state':
            path_to_screenshot: str = os.path.join(path_to_screenshots_dir, os.path.basename(item['data']['path_to_screenshot']))
            screenshot_base64: str = encode_image(path_to_screenshot)
            prompt_s_a_sequence.append({
                "role": "user", 
                "path_to_screenshot" : path_to_screenshot,
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{screenshot_base64}"
                    },
                }],
            })
            if is_act and is_crop_to_action and item_idx + 1 < len(trace_json):
                # If we want to crop the screenshot to where the action is located,
                # we kept a trace of actions, and this isn't the last state (i.e. an action is taken from it)...
                img = Image.open(path_to_screenshot)
                path_to_cropped_img: str = os.path.join(path_to_output_dir, f"cropped_{os.path.basename(path_to_screenshot)}")
                # Find the last (inclusive) action with coordinates
                next_action = trace_json[item_idx + 1]
                if next_action['data']['type'] in ['click', 'mousedown', 'mouseup']:
                    next_action_x_coord, next_action_y_coord = next_action['data']['x'], next_action['data']['y']
                    # Crop the image to where the action is located
                    crop_coords = (next_action_x_coord - crop_box_width // 2, # top_left_x
                                next_action_y_coord - crop_box_width // 2, # top_left_y
                                next_action_x_coord + crop_box_width // 2, # bottom_right_x
                                next_action_y_coord + crop_box_width // 2) # bottom_right_y

                    # NOTE: Due to Mac Retina display, the screenshot is 2x the size of the actual screen
                    # So, we need to rescale the coordinates to match the actual screen size
                    crop_coords = tuple([coord * 2 for coord in crop_coords])
                    cropped_img = img.crop(crop_coords) # creates a `crop_box_width` x `crop_box_width` box sub-image around action coordinates
                    buffer = BytesIO()
                    cropped_img.save(buffer, format="PNG")
                    cropped_img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    cropped_img.save(path_to_cropped_img)
                elif next_action['data']['type'] in ['keypress', 'keystroke']:
                    # use entire screen for keypresses b/c we don't know where on the screen the text field is focused
                    # we can't rely on the coordinates of the previous click b/c the focusable element for the text field might be huge
                    # and the click might be on the edge of the element, so we'd miss the actual text field
                    # TODO: Use OCR to find the text field and crop to that
                    cropped_img_base64 = screenshot_base64
                    shutil.copy(path_to_screenshot, path_to_cropped_img)
                elif next_action['data']['type'] in ['scroll']:
                    # use entire screen for scrolls
                    cropped_img_base64 = screenshot_base64
                    shutil.copy(path_to_screenshot, path_to_cropped_img)
                else:
                    raise Exception(f"Unknown action type for `next_action` in `s_a_sequence`: {next_action['data']['type']}")
                
                cropped_images_base64.append({
                    "role": "user", 
                    "path_to_screenshot" : path_to_cropped_img,
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{cropped_img_base64}"
                        },
                    }],
                })
        elif item['type'] == 'action' and is_act:
            action: str = convert_trace_action_to_dsl(item, is_filtered_action_subset=False)['action']
            prompt_s_a_sequence.append({
                "role": "assistant", 
                "content": [{
                    "type" : "text",
                    "text" : f"Action: {action}",
                }],
            })
        else:
            raise Exception(f"Unknown type for `item` in `s_a_sequence`: {type(item)}")

    if is_crop_to_action:
        assert len(prompt_s_a_sequence) - 1 == len(cropped_images_base64) * 2, f"Expected len(prompt_s_a_sequence) -1 == len(cropped_images_base64) * 2, got {len(prompt_s_a_sequence) - 1} != {len(cropped_images_base64) * 2}"

    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : (
                prompt__td_kf_act_intro__pairwise(task_descrip, ui_name) if (is_pairwise and is_crop_to_action) else 
                prompt__td_kf_act_intro__pairwise__cropped(task_descrip, ui_name) if is_pairwise else
                prompt__td_kf_act_intro(task_descrip, ui_name)
            )
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__td_kf_act_close__pairwise() if is_pairwise else prompt__td_kf_act_close()
        }]
    }

    if is_pairwise:
        # if is_act=True, then feed (S, A, S') -- i.e. a pair of before/after screenshots with intermediate action
        # if is_act=False, then feed (S, S') -- i.e. a pair of before/after screenshots
        responses: List[str] = []
        is_act_adjustment: int = 1 if is_act else 0 # If we're including actions, then we need to adjust the loop to account for the extra action
        step_size: int = 1 + is_act_adjustment
        
        for i in tqdm(range(0, len(prompt_s_a_sequence)- step_size, step_size)):
            # Create messages
            if is_crop_to_action:
                assert is_act, "If is_crop_to_action=True, then is_act must also be True"
                cropped_img = cropped_images_base64[i//step_size]
                messages: List[str] = [intro_prompt] + [ prompt_s_a_sequence[i], prompt_s_a_sequence[i+1], cropped_img, prompt_s_a_sequence[i+2] ] + [close_prompt]
                assert len(messages) == 1 + 1 + step_size + 1 + 1, f"Expected {1+1+step_size+1+1} prompts, got {len(messages)}"
            else:
                messages: List[str] = [intro_prompt] + prompt_s_a_sequence[i:i+1+step_size] + [close_prompt]
                assert len(messages) == 1 + 1 + step_size + 1, f"Expected {1+1+step_size+1} prompts, got {len(messages)}"
            
            # Fetch completion
            try:
                new_messages = [ { key: val for key,val in m.items() if key != 'path_to_screenshot' } for m in messages if m is not None]
                response: str = _fetch_openai_completion(new_messages, model='gpt-4-vision-preview', temperature=0.0)
            except Exception as e:
                print(f"Error for task_descrip={task_descrip} | demo_name={demo_name} | i={i}: {e}")
                raise e
            responses.append(response)
        response: str = "\n>>>>>>>>>>>\n".join(responses)
    else:
        # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
        messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
        try:
            response: str = _fetch_openai_completion(messages, model='gpt-4-vision-preview', temperature=0.0)
        except Exception as e:
            print(f"Error for task_descrip={task_descrip} | demo_name={demo_name}: {e}")
            raise e

    # Remove extraneous chars
    response = response.replace("```\n", "").replace("```", "")
    
    # Ablation name
    ablation = ' -- '.join([
        "pairwise=true" if is_pairwise else "pairwise_false",
        "is_act=true" if is_act else "is_act_false",
    ])
    
    # If pairwise, then save intermediary pairwise representation, then have LLM piece together SOP
    sop_base_name: str = f"Generated-SOP -- {demo_name} -- {ablation}.txt"
    if is_pairwise:
        with open( os.path.join(path_to_output_dir, f"[temp] {sop_base_name}"), "w" ) as f:
            f.write(f"Task: {task_descrip}\n")
            f.write("----------------------------------------\n")
            f.write(response)
        messages: List[str] = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": prompt__join_pairwise(response, '>>>>>>>>>>>'),
            }],
        }]
        response: str = _fetch_openai_completion(messages, model='gpt-4', temperature=0.0)
    
    # Save SOP
    with open( os.path.join(path_to_output_dir, f"[raw] {sop_base_name}"), "w" ) as f:
        f.write(f"Task: {task_descrip}\n")
        f.write("----------------------------------------\n")
        f.write(response)

    # Create generalized version of SOP
    messages: List[str] = [{
        "role": "system",
        "content": [{
            "type": "text",
            "text": prompt__generalize(response, task_descrip),
        }],
    }]
    response: str = _fetch_openai_completion(messages, model='gpt-4', temperature=0.0)
    with open( os.path.join(path_to_output_dir, f"[general] {sop_base_name}"), "w" ) as f:
        f.write(f"Task: {task_descrip}\n")
        f.write("----------------------------------------\n")
        f.write(response)
    
    return response

if __name__ == "__main__":
    args = parse_args()
    path_to_demo_folder: str = args.path_to_demo_folder
    task_descrip: Optional[str] = args.task_descrip
    is_pairwise: bool = args.is_pairwise
    is_act: bool = args.is_act
    is_crop_to_action: bool = args.is_crop_to_action
    
    if is_crop_to_action:
        assert is_act, "If is_crop_to_action=True, then is_act must also be True"

    generate_sop_from_demo(
        path_to_demo_folder, 
        task_descrip,
        is_act,
        is_pairwise,
        is_crop_to_action,
    )