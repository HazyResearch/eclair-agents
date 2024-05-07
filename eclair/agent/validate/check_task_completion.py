"""
Usage:

python main.py <PATH_TO_TASKS_DIR> FLAGS
"""

import traceback
from typing import Any, Dict, List, Optional
import pandas as pd
import os
from workflows.helpers import (
    _fetch_completion,
    build_prompt_s_a_sequence,
    add_standard_experiment_args,
    get_folders_for_task_id,
    get_path_to_screenshots_dir,
    get_path_to_sop_txt,
    get_path_to_trace_json,
)
import argparse
import json
from eclair.hospital_demo.validate.prompts import (
    prompt__validate_task_completion__intro,
    prompt__validate_task_completion__close,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_standard_experiment_args(parser)
    parser.add_argument(
        "--is_include_sop",
        action="store_true",
        help="If TRUE, include SOP.txt in model input",
    )
    parser.add_argument(
        "--is_td",
        action="store_true",
        default=False,
        help="If TRUE, then include task description into prompts",
    )
    parser.add_argument(
        "--is_kf",
        action="store_true",
        default=False,
        help="If TRUE, then include screenshots as key frames into prompts",
    )
    parser.add_argument(
        "--is_act",
        action="store_true",
        default=False,
        help="If TRUE, then include action traces into prompts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GPT4",
        help="Model to use for self monitoring, one of [GPT4, GeminiPro]",
        choices=["GPT4", "GeminiPro"],
    )
    parser.add_argument(
        "--is_verbose", action="store_true", help="If TRUE, then print out stuff"
    )
    return parser.parse_args()


def helper_task_completion(
    gt_trace: Dict[str, Any],
    task_descrip: str,
    model: str,
    sop: str,
    gt_is_met: bool,
    path_to_screenshots: str,
    is_td: bool,
    is_kf: bool,
    is_act: bool,
    task_type: str,
) -> Dict[str, str]:
    """Helper fx to eval a single POSITIVE or NEGATIVE example."""
    prompt_s_a_sequence, paths_to_screenshots = build_prompt_s_a_sequence(
        gt_trace, path_to_screenshots
    )
    if not is_td:
        task_descrip = None
    if not is_kf:
        prompt_s_a_sequence = [
            x for x in prompt_s_a_sequence if x["content"][0]["type"] != "image_url"
        ]  # remove screenshots
        paths_to_screenshots = []
    if not is_act:
        prompt_s_a_sequence = [
            x for x in prompt_s_a_sequence if x["content"][0]["type"] != "text"
        ]  # remove actions

    intro_prompt: Dict[str, str] = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt__validate_task_completion__intro(task_descrip, sop),
            }
        ],
    }
    close_prompt: Dict[str, str] = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt__validate_task_completion__close()}
        ],
    }
    # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
    pred_raw_response: str = _fetch_completion(messages, model)

    # Evaluate
    try:
        pred_json = json.loads(
            pred_raw_response.replace("```json", "").replace("```", "").strip()
        )
        pred_rationale: Dict[str, str] = pred_json["thinking"]
        pred_is_met: bool = pred_json["was_completed"]
        is_correct: bool = pred_is_met == gt_is_met
    except:
        pred_rationale = None
        pred_is_met = None
        is_correct = False

    return {
        # gt
        "gt_is_met": gt_is_met,
        "paths_to_screenshots": paths_to_screenshots,
        # preds
        "pred_rationale": pred_rationale,
        "pred_is_met": pred_is_met,
        "pred_raw_response": pred_raw_response,
        # eval
        "is_correct": is_correct,
        "task_type": task_type,
        "model": model,
    }


def validate_task_completion(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model: str,
    sop: Optional[str],
    is_td: bool,
    is_kf: bool,
    is_act: bool,
) -> pd.DataFrame:
    """
    Evaluate overall task completion success.
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_descrip: str = "Place an order for a sitter."
    results: List[Dict[str, Any]] = []

    results.append(
        helper_task_completion(
            gt_trace,
            task_descrip,
            model,
            sop,
            True,
            path_to_screenshots,
            is_td,
            is_kf,
            is_act,
            task_type="true",
        )
    )

    df = pd.DataFrame(results)
    return df


def kwarg_setting_to_ablation(kwarg_setting: Dict[str, Any]) -> str:
    # Parse kwargs
    is_include_sop: bool = kwarg_setting["is_include_sop"]
    is_td: bool = kwarg_setting["is_td"]
    is_kf: bool = kwarg_setting["is_kf"]
    is_act: bool = kwarg_setting["is_act"]
    model: str = kwarg_setting["model"]
    # Generate ablation string
    short_name: str = model
    if is_include_sop:
        short_name += "_sop"
    if is_td:
        short_name += "_td"
    if is_kf:
        short_name += "_kf"
    if is_act:
        short_name += "_act"
    return short_name


def run(
    path_to_demo_folder: str,
    path_to_output_dir: str,
    task_id: int,
    model: str,
    is_include_sop: bool,
    is_td: bool,
    is_kf: bool,
    is_act: bool,
    is_verbose: bool = False,
):
    # Create output directory
    path_to_output_dir: str = (
        "/Users/avanikanarayan/Developer/Research/demonstration-collection/data/experimental_results"
    )
    demo_name: str = "Order a sitter order"
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load files
    path_to_sop_file: str = (
        "/Users/avanikanarayan/Developer/Research/demonstration-collection/data/[new] ECLAIR - Order a sitter order @ 2024-03-30-16-09-01/manual_sop_sitter.txt"  # get_path_to_trace_json(path_to_demo_folder)
    )
    path_to_screenshots_dir: str = (
        "/Users/avanikanarayan/Developer/Research/demonstration-collection/data/[new] ECLAIR - Order a sitter order @ 2024-03-30-16-09-01/screenshots"  # get_path_to_screenshots_dir(path_to_demo_folder)
    )
    path_to_trace: str = (
        "/Users/avanikanarayan/Developer/Research/demonstration-collection/data/[new] ECLAIR - Order a sitter order @ 2024-03-30-16-09-01/Order a sitter order @ 2024-05-04-17-27-41.json"  # get_path_to_sop_txt(path_to_demo_folder)
    )

    # Read files
    trace_json: Dict[str, Any] = json.load(open(path_to_trace, "r"))
    sop: str = open(path_to_sop_file, "r").read() if is_include_sop else None

    states = trace_json["log"]["states"]
    actions = trace_json["log"]["actions"]

    # interleave states and actions
    interleaved = []
    for i in range(len(states) + len(actions)):
        if i % 2 == 0:
            state = states.pop(0)
            state["type"] = "state"
            state["data"] = {}
            state["data"]["id"] = i

            interleaved.append(state)
        else:
            action = actions.pop(0)
            action["type"] = "action"
            action["data"] = {}
            action["data"]["element_attributes"] = None
            action["data"]["id"] = i
            interleaved.append(action)

    trace_json["trace"] = interleaved

    # Execute eval
    try:
        df = validate_task_completion(
            trace_json,
            path_to_screenshots_dir,
            model,
            sop,
            is_td=is_td,
            is_kf=is_kf,
            is_act=is_act,
        )
    except Exception as e:
        print(f"Error with demo folder: {path_to_demo_folder} | task_id: {task_id}")
        print(traceback.format_exc())
        print(str(e))
        raise e

    # Ablation
    ablation = kwarg_setting_to_ablation(
        {
            "is_include_sop": is_include_sop,
            "is_td": is_td,
            "is_kf": is_kf,
            "is_act": is_act,
            "model": model,
        }
    )
    df["demo_name"] = demo_name
    df["task_id"] = task_id
    df["ablation--is_td"] = is_td
    df["ablation--is_kf"] = is_kf
    df["ablation--is_act"] = is_act
    df["ablation--is_include_sop"] = is_include_sop
    df["ablation--model"] = model

    # Print metrics
    accuracy: float = df["is_correct"].mean() if "is_correct" in df.columns else "N/A"
    all_correct: bool = df["is_correct"].all() if "is_correct" in df.columns else "N/A"
    if is_verbose:
        print(f"Task: {task_id}")
        print(f"Accuracy: {accuracy}")
        print(f"All correct? {all_correct}")
    df.to_csv(
        os.path.join(path_to_output_dir, f"self_monitoring__{ablation}.csv"),
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    task_id: bool = args.task_id

    # Task-specific flags
    is_include_sop: bool = args.is_include_sop
    is_td: bool = args.is_td
    is_kf: bool = args.is_kf
    is_act: bool = args.is_act
    model: str = args.model

    # Identify folders corresponding to this `task_id` in `path_to_input_dir`
    path_to_demo_folders: List[str] = get_folders_for_task_id(
        path_to_input_dir, task_id
    )

    # Loop through each demos's folder...
    # for path_to_demo_folder in path_to_demo_folders:
    breakpoint()
    run(
        "",
        path_to_output_dir,
        task_id,
        model,
        is_td,
        is_kf,
        is_act,
        is_include_sop,
    )
