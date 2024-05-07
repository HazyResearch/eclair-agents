"""
Usage:

python evaluate.py --is_actuation --path_to_task_dir "/Users/avanikanarayan/Developer/Research/eclair-agents/eclair/vldb_experiments/[VLDB] 30 WebArena Tasks/104 @ 2023-12-30-12-23-12"
"""

from tqdm import tqdm
from typing import Any, Dict, List
import pandas as pd
import os
from eclair.utils.helpers import (
    adjust_json_state_xy_coords_to_center,
    extract_coords_from_dsl_CLICK,
    extract_coords_from_dsl_SCROLL,
    extract_text_from_dsl_PRESS,
    extract_text_from_dsl_TYPE,
    is_coords_within_element,
    load_screenshot_for_state,
    load_files_for_task
)
from typing import Dict, List, Optional
from eclair.vldb_experiments.execute_actions.actuation_selector import (
    ActuationSelector,
)
from eclair.vldb_experiments.execute_actions.action_selector import (
    ActionSelectorL1,
)
import argparse
import json
import numpy as np
import dirtyjson


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_task_dir", type=str, required=True, help="Path to task demo folder"
    )
    parser.add_argument(
        "--is_action",
        action="store_true",
        help="If TRUE, eval next action suggestion given a gt previous actions.",
    )
    parser.add_argument(
        "--is_actuation",
        action="store_true",
        help="If TRUE, eval actuation given a gt next action suggestion.",
    )
    parser.add_argument(
        "--is_include_sop",
        action="store_true",
        help="If TRUE, include SOP.txt in model input",
    )
    return parser.parse_known_args()


############################################
############################################
#
# Correctness check helpers
#
############################################
############################################


def get_action_type_from_dsl(dsl: str, action_type: str) -> Optional[str]:
    """Actuation string `dsl` could have multiple discrete actuations (e.g. "CLICK(1,2)|TYPE('hello')|SCROLL(1)").
    This returns the first occurrence of the action type we're looking for."""
    # Find TYPE action in `dsl`, which could have multiple actions
    dsl: str = [x for x in dsl.split("|") if x.lower().startswith(action_type.lower())]
    if len(dsl) == 0:
        return None
    return dsl[0]


def is_click_correct(
    gt_actuation: str,
    gt_element: Dict[str, str],
    pred_actuation: str,
    pred_element: Dict[str, str],
) -> bool:
    """CLICK is correct if...
    - pred_actuation (x,y) is within the gt_element's bbox
    """
    # Find CLICK action in pred
    pred_actuation = get_action_type_from_dsl(pred_actuation, "CLICK")
    if not pred_actuation:
        return False

    x_coord, y_coord = extract_coords_from_dsl_CLICK(pred_actuation)

    # Eval
    if is_coords_within_element(x_coord, y_coord, gt_element):
        return True
    return False


def is_scroll_correct(
    gt_actuation: str,
    gt_element: Dict[str, str],
    pred_actuation: str,
    pred_element: Dict[str, str],
) -> bool:
    """SCROLL is correct if...
    - sign(gt_dy) == sign(pred_dy)
    """
    # Find SCROLL action in pred
    pred_actuation = get_action_type_from_dsl(pred_actuation, "SCROLL")
    if not pred_actuation:
        return False

    pred_dy: float = extract_coords_from_dsl_SCROLL(pred_actuation)
    gt_dy: float = extract_coords_from_dsl_SCROLL(gt_actuation)

    # Eval
    if (gt_dy < 0 and pred_dy < 0) or (gt_dy > 0 and pred_dy > 0):
        return True
    return False


def is_type_correct(
    gt_actuation: str,
    gt_element: Dict[str, str],
    pred_actuation: str,
    pred_element: Dict[str, str],
) -> bool:
    """TYPE is correct if...
    - pred text == gt text
    """
    # Find TYPE action in pred
    pred_actuation = get_action_type_from_dsl(pred_actuation, "TYPE")
    if not pred_actuation:
        return False

    pred_text: str = extract_text_from_dsl_TYPE(pred_actuation)
    gt_text: str = extract_text_from_dsl_TYPE(gt_actuation)

    # Eval
    if gt_text.lower().strip() == pred_text.lower().strip():
        return True
    return False


def is_press_correct(
    gt_actuation: str,
    gt_element: Dict[str, str],
    pred_actuation: str,
    pred_element: Dict[str, str],
) -> bool:
    """PRESS is correct if...
    - pred text == gt text (plus or minus similar names, e.g. return/enter)
    """
    # Find PRESS action in pred
    pred_actuation = get_action_type_from_dsl(pred_actuation, "PRESS")
    if not pred_actuation:
        return False

    pred_text: str = extract_text_from_dsl_PRESS(pred_actuation)
    gt_text: str = extract_text_from_dsl_PRESS(gt_actuation)

    # Eval
    return_equivalents: List[str] = ["return", "enter"]
    if gt_text == pred_text:
        return True
    elif (
        gt_text.lower() in return_equivalents
        and pred_text.lower() in return_equivalents
    ):
        return True
    return False


def is_actuation_correct(
    gt_actuation: str,
    gt_element: Dict[str, str],
    pred_actuation: str,
    pred_element: Dict[str, str],
) -> bool:
    for actuation in pred_actuation.split("|"):
        actuation = actuation.strip()
        if gt_actuation.startswith("CLICK"):
            if is_click_correct(gt_actuation, gt_element, actuation, pred_element):
                return True
        elif gt_actuation.startswith("TYPE"):
            if is_type_correct(gt_actuation, gt_element, actuation, pred_element):
                return True
        elif gt_actuation.startswith("PRESS"):
            if is_press_correct(gt_actuation, gt_element, actuation, pred_element):
                return True
        elif gt_actuation.startswith("SCROLL"):
            if is_scroll_correct(gt_actuation, gt_element, actuation, pred_element):
                return True
    return False


def execute_actuation(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model_kwargs: Dict[str, str],
    sop: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate actuation.
    Injects gt next action suggestion into model and compares its predicted actuation with gt actuation.
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_description: str = gt_task_data["webarena"]["intent"]
    task_id: int = gt_task_data["webarena"]["task_id"]
    actuation_selector: ActuationSelector = ActuationSelector(model_kwargs=model_kwargs)

    # group task pipeline into sets of two
    previous_actions: List[str] = []
    results: List[Dict[str, Any]] = []
    for idx in tqdm(
        range(0, len(gt_trace) - 1, 2), desc=f"Task: {task_id}"
    ):  # -1 because last state has not following action
        state: Dict[str, str] = gt_trace[idx]
        action: Dict[str, str] = gt_trace[idx + 1]

        # Load screenshot
        path_to_screenshot, encoded_image = load_screenshot_for_state(
            state, path_to_screenshots
        )

        # Load inputs
        next_action_suggestion: str = state["gt_labels"]["next_action_suggestion"]
        sop_step: str = state["gt_labels"]["sop_instruction"]
        json_state: Dict[str, str] = json.loads(state["data"]["json_state"])
        gt_element = (
            action["data"].get("element_attributes", {}).get("element", {}).copy()
        )
        random_loc = np.random.randint(0, len(json_state))
        json_state.insert(random_loc, gt_element)
        # json_state.append(gt_element)
        json_state = adjust_json_state_xy_coords_to_center(json_state)

        # Run model
        filtered_json_state, pred_raw_response = actuation_selector.run(
            previous_actions=previous_actions,
            action_suggestion=next_action_suggestion,
            json_state=json_state,
            task=task_description,
            screen_shot=encoded_image,
            is_vision=True,
            sop_step=sop_step,
        )
        previous_actions.append(next_action_suggestion)
        gt_actuation: str = state["gt_labels"]["actuation"]
        gt_element = action["data"].get("element_attributes", {}).get("element", {})
        # Evaluate
        try:
            pred_json = dirtyjson.loads(
                pred_raw_response.replace("```json", "").replace("```", "").strip()
            )
            pred_element: Dict[str, str] = pred_json["element"]
            # pred_actuation: str = pred_json["action"].replace(" ", "")
            pred_actuation = pred_json["action"]
            is_correct: bool = is_actuation_correct(
                gt_actuation, gt_element, pred_actuation, pred_element
            )

        except:
            pred_actuation = None
            pred_element = None
            pred_actuation = None
            is_correct = False

        # Save results
        results.append(
            {
                # metadata
                "state_id": state["data"]["id"],
                "action_id": action["data"]["id"],
                "path_to_screenshot": path_to_screenshot,
                # gt
                "gt_actuation": gt_actuation,
                "gt_element": gt_element,
                "gt_action_type": action["gt_labels"]["action_type"],
                # preds
                "pred_actuation": pred_actuation,
                "pred_element": pred_element,
                "pred_raw_response": pred_raw_response,
                "filtered_json_state": filtered_json_state,
                # eval
                "is_correct": is_correct,
            }
        )

    df = pd.DataFrame(results)
    return df


def execute_next_action(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model_kwargs: Dict[str, str],
    sop: Optional[str] = None,
) -> pd.DataFrame:
    """
    Evaluate action suggestion.
    Injects gt previous actions into model and compares its predicted next action suggestion with gt next action suggestion.
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_description: str = gt_task_data["webarena"]["intent"]
    task_id: int = gt_task_data["webarena"]["task_id"]
    action_selector: ActuationSelector = ActionSelectorL1(
        model_kwargs=model_kwargs, task_outline=sop
    )

    # group task pipeline into sets of two
    previous_actions: List[str] = []
    results: List[Dict[str, Any]] = []
    for idx in tqdm(
        range(0, len(gt_trace) - 1, 2), desc=f"Task: {task_id}"
    ):  # -1 because last state has not following action
        state: Dict[str, str] = gt_trace[idx]
        action: Dict[str, str] = gt_trace[idx + 1]

        # Load screenshot
        path_to_screenshot, encoded_image = load_screenshot_for_state(
            state, path_to_screenshots
        )

        # Load inputs
        json_state: Dict[str, str] = json.loads(state["data"]["json_state"])
        json_state = adjust_json_state_xy_coords_to_center(json_state)

        # Run model
        pred_raw_response: str = action_selector.run(
            previous_actions=previous_actions,
            json_state=json_state,
            task=task_description,
            screen_shot=encoded_image,
            is_vision=True,
        )

        # Evaluate
        gt_next_action_suggestion: str = state["gt_labels"]["next_action_suggestion"]
        previous_actions.append(gt_next_action_suggestion)

        # Evaluate
        try:
            pred_json = dirtyjson.loads(
                pred_raw_response.replace("```json", "").replace("```", "").strip()
            )
            pred_rationale: Dict[str, str] = pred_json["rationale"]
            pred_next_action_suggestion: str = pred_json["action"]
        except:
            pred_json = None
            pred_rationale = None
            pred_next_action_suggestion = None

        # Save results
        results.append(
            {
                # metadata
                "state_id": state["data"]["id"],
                "action_id": action["data"]["id"],
                "path_to_screenshot": path_to_screenshot,
                # gt
                "gt_next_action_suggestion": gt_next_action_suggestion,
                "gt_action_type": action["gt_labels"]["action_type"],
                # preds
                "pred_rationale": pred_rationale,
                "pred_next_action_suggestion": pred_next_action_suggestion,
            }
        )

    df = pd.DataFrame(results)
    return df


def main(args):
    path_to_task_dir: str = args.path_to_task_dir
    is_actuation: bool = args.is_actuation
    is_action: bool = args.is_action
    is_include_sop: bool = args.is_include_sop

    # Output dir
    path_to_output_dir: str = os.path.join(path_to_task_dir, "experiments/")
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Find input files
    inputs = load_files_for_task(path_to_task_dir, is_include_sop)

    path: str = os.path.join(
        path_to_output_dir,
        f"execute_{'action' if is_action else 'actuation'}_sop.csv",
    )

    if os.path.exists(path):
        # Skip if output already exists
        return

    if is_actuation:
        # Actuation
        df = execute_actuation(
            inputs.gt_task_data,
            inputs.path_to_screenshots,
            inputs.model_kwargs,
            inputs.sop,
        )
    elif is_action:
        # Next action suggestion
        print("Action")
        df = execute_next_action(
            inputs.gt_task_data,
            inputs.path_to_screenshots,
            inputs.model_kwargs,
            inputs.sop,
        )
    else:
        raise ValueError(f"Must specify either --is_actuation or --is_action")

    # Print metrics
    accuracy: float = df["is_correct"].mean() if "is_correct" in df.columns else "N/A"
    all_correct: bool = df["is_correct"].all() if "is_correct" in df.columns else "N/A"

    print(f"Task: {path_to_task_dir}")
    print(f"Accuracy: {accuracy}")
    print(f"All correct? {all_correct}")

    # dump accuracy to .txt file
    with open(os.path.join(path_to_output_dir, "accuracy_action.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"All correct? {all_correct}\n")

    df.to_csv(
        os.path.join(
            path_to_output_dir,
            f"execute_{'action' if is_action else 'actuation'}_nosop.csv",
        )
    )


if __name__ == "__main__":
    args, __ = parse_args()
    main(args)
