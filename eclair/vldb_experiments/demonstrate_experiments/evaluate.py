import os
from eclair.utils.helpers import (
    _fetch_openai_completion,
    convert_trace_action_to_dsl,
    encode_image,
    fetch_openai_vision_completion,
    get_webarena_task_json,
)
from typing import Dict, List, Optional
import json
import argparse
from eclair.vldb_experiments.demonstrate_experiments.prompts import (
    prompt__td,
    prompt__td_kf,
    prompt__td_kf__pairwise,
    prompt__td_kf_act_intro,
    prompt__td_kf_act_close,
    prompt__td_kf_act_intro__pairwise,
    prompt__td_kf_act_close__pairwise,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "path_to_task_dir", help="Path to task folder. Should have a subfolder named `screenshots/`, there should be a `trace.json` file", )
    parser.add_argument( "--path_to_output_dir", default="./demonstrate_outputs/", type=str, required=False, help="Path to output directory", )
    parser.add_argument( "--is_td", action="store_true", default=False, help="If TRUE, then include task description into prompts",)
    parser.add_argument( "--is_td_kf", action="store_true", default=False, help="If TRUE, then include screenshots as key frames into prompts", )
    parser.add_argument( "--is_td_kf_act", action="store_true", default=False, help="If TRUE, then include action trace as prompts" )
    parser.add_argument( "--is_pairwise", action="store_true", default=False, help="If TRUE, then instead of prompting all screenshots / action traces at once, prompt two at a time (before/after), and piece together SOP afterwards" )
    return parser.parse_args()

def find_files_by_prefix_suffix(directory, prefix="SOP", suffix=".txt") -> Optional[str]:
    matching_files = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(suffix):
            matching_files.append(file)
    return matching_files if len(matching_files) > 0 else None

if __name__ == "__main__":
    args = parse_args()
    path_to_task_dir: str = args.path_to_task_dir
    is_td: bool = args.is_td
    is_td_kf: bool = args.is_td_kf
    is_td_kf_act: bool = args.is_td_kf_act
    is_pairwise: bool = args.is_pairwise
    path_to_screenshots_dir: str = os.path.join(path_to_task_dir, f"screenshots/")
    assert is_td + is_td_kf + is_td_kf_act == 1, "Must specify exactly one of --is_td, --is_td_kf, --is_td_kf_act"

    # Get WebArena task description
    trace_name: str = os.path.basename(path_to_task_dir.strip("/"))
    task_id: int = int(trace_name.split(" @ ")[0])
    task_json: Optional[Dict[str, str]] = get_webarena_task_json(task_id)
    assert task_json is not None, f"Could not find WebArena task json for {trace_name}"
    task_descrip: str = task_json["intent"]
    start_url: str = task_json["start_url"]
    ui_name: str = {
        'gitlab' : 'Gitlab',
        'shopping' : 'Generic e-commerce site',
        'shopping_admin' : 'Generic e-commerce admin based on Adobe Magneto',
        'reddit' : 'Generic open source Reddit clone',
    }[task_json["sites"][0]]

    # Get .json trace
    path_to_trace: str = os.path.join(path_to_task_dir, f"{trace_name}.json")
    
    # Create output directory
    path_to_output_dir: str = os.path.join(args.path_to_output_dir, trace_name)
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # If ground truth SOP.txt exists in input_dir, then copy over into output_dir
    sop_files = find_files_by_prefix_suffix(path_to_task_dir, "SOP", ".txt")
    if sop_files is not None:
        assert len(sop_files) == 1, f"Expected 1 SOP file, got {len(sop_files)}"
        path_to_sop_file: str = os.path.join(path_to_task_dir, sop_files[0])
        gt_sop = open(path_to_sop_file, "r").read()
        with open(os.path.join(path_to_output_dir, sop_files[0]), 'w') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write(f"Task: {task_descrip}\n")
            f.write(f"UI: {ui_name}\n")
            f.write("----------------------------------------\n")
            f.write(gt_sop)

    if is_td:
        assert is_pairwise is False, "Pairwise not supported for TD"
        print("TD")
        prompt: str = prompt__td(task_descrip, ui_name)
        response: str = fetch_openai_vision_completion(
            prompt,
            [],
            temperature=0.0,
        )
    elif is_td_kf:
        print(f"TD + KF {'pairwise' if is_pairwise else ''}")
        screenshots: List[str] = sorted(os.listdir(path_to_screenshots_dir)) # sort by filename, from 0 -> N
        base64_screenshots: List[str] = [
            encode_image(os.path.join(path_to_screenshots_dir, x)) for x in screenshots
        ]
        if is_pairwise:
            # Feed (S, S') -- i.e. a pair of before/after screenshots
            responses: List[str] = []
            for i in range(0, len(screenshots)-1):
                assert len(base64_screenshots[i:i+2]) == 2, f"Expected 2 screenshots, got {len(base64_screenshots[i:i+2])}"
                prompt: str = prompt__td_kf__pairwise(task_descrip, ui_name)
                response: str = fetch_openai_vision_completion(
                    prompt,
                    base64_screenshots[i:i+2],
                    temperature=0.0,
                )
                responses.append(response)
            response: str = "\n>>>>>>>>>>>\n".join(responses)
        else:
            # Feed (S, S', S'', ...) -- i.e. all screenshots at once
            prompt: str = prompt__td_kf(task_descrip, ui_name)
            response: str = fetch_openai_vision_completion(
                prompt,
                base64_screenshots,
                temperature=0.0,
            )
    elif is_td_kf_act:
        print(f"TD + KF + ACT {'pairwise' if is_pairwise else ''}")
        screenshots: List[str] = os.listdir(path_to_screenshots_dir)
        screenshots = sorted(screenshots) # sort by filename, from 0 -> N
        base64_screenshots: List[str] = [
            encode_image(os.path.join(path_to_screenshots_dir, x)) for x in screenshots
        ]
        # Loop through trace, interleaving screenshots (states) and actions
        prompt_s_a_sequence: List[str] = []
        trace_json: Dict[str, str] = json.loads(open(path_to_trace, "r").read())['trace']
        for item in trace_json:
            if item['type'] == 'state':
                path_to_screenshot: str = os.path.join(path_to_screenshots_dir, os.path.basename(item['data']['path_to_screenshot']))
                screenshot_base64: str = encode_image(path_to_screenshot)
                prompt_s_a_sequence.append({
                    "role": "user", 
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{screenshot_base64}"
                        },
                    }],
                })
            elif item['type'] == 'action':
                action: str = convert_trace_action_to_dsl(item)['action']
                prompt_s_a_sequence.append({
                    "role": "assistant", 
                    "content": [{
                        "type" : "text",
                        "text" : f"Action: {action}",
                    }],
                })
            else:
                raise Exception(f"Unknown type for `item` in `s_a_sequence`: {type(item)}")
        intro_prompt: Dict[str, str] = {
            "role" : "user",
            "content" : [{
                "type" : "text",
                "text" : prompt__td_kf_act_intro__pairwise(task_descrip, ui_name) if is_pairwise else prompt__td_kf_act_intro(task_descrip, ui_name)
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
            # Feed (S, A, S') -- i.e. a pair of before/after screenshots
            responses: List[str] = []
            for i in range(0, len(prompt_s_a_sequence)-2, 2):
                messages: List[str] = [intro_prompt] + prompt_s_a_sequence[i:i+3] + [close_prompt]
                assert len(messages) == 1 + 3 + 1, f"Expected 5 prompts, got {len(messages)}"
                response: str = _fetch_openai_completion(messages, model='gpt-4-vision-preview', temperature=0.0)
                responses.append(response)
            response: str = "\n>>>>>>>>>>>\n".join(responses)
        else:
            # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
            messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
            response: str = _fetch_openai_completion(messages, model='gpt-4-vision-preview', temperature=0.0)
    else:
        raise ValueError("Must specify at least one of --is_td, --is_td_kf, --is_td_kf_act")

    print(response)
    
    # Remove extraneous chars
    response = response.replace("```\n", "").replace("```", "")

    # Save SOP
    short_name: str = 'td' if is_td else 'td_kf' if is_td_kf else 'td_kf_act'
    short_name += '__pairwise' if is_pairwise else ''
    with open( os.path.join(path_to_output_dir, f"GPT4V-SOP - {short_name} - {trace_name}.txt"), "w" ) as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Task: {task_descrip}\n")
        f.write(f"UI: {ui_name}\n")
        f.write(f"Start URL: {start_url}\n")
        f.write("----------------------------------------\n")
        f.write(response)
