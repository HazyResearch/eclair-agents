"""
Usage:

python evaluate.py "/Users/mwornow/Downloads/test/104 @ 2023-12-30-12-23-12" --is_task_trajectory
"""
import random
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import pandas as pd
import os
from eclair.utils.helpers import (
    _fetch_openai_completion,
    convert_trace_action_to_dsl,
    fetch_openai_vision_completion,
    load_screenshot_for_state,
    load_files_for_task
)
from typing import Dict, List, Optional
import argparse
import json
from prompts import (
    prompt__validate_condition, 
    prompt__validate_actuation,
    prompt__validate_task_completion__intro,
    prompt__validate_task_completion__close,
    prompt__validate_task_trajectory__intro,
    prompt__validate_task_trajectory__close,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_task_dir", type=str, help="Path to task demo folder")
    parser.add_argument( "--path_to_output_dir", default="./outputs/", type=str, required=False, help="Path to output directory", )
    parser.add_argument("--is_actuation", action="store_true", help="If TRUE, then eval on actuation")
    parser.add_argument("--is_precondition", action="store_true", help="If TRUE, then eval on preconditions")
    parser.add_argument("--is_task_completion", action="store_true", help="If TRUE, then eval on task completion success")
    parser.add_argument("--is_task_trajectory_valid", action="store_true", help="If TRUE, then eval on task trajectory accuracy")
    parser.add_argument("--is_include_sop", action="store_true", help="If TRUE, include SOP.txt in model input")
    return parser.parse_known_args()

def helper_precondition(state: Dict[str, Any], action: Dict[str, Any], gt_precondition: str, gt_is_met: bool, path_to_screenshots: str) -> Dict[str, str]:
    """Helper fx to eval a single POSITIVE or NEGATIVE example."""
    path_to_screenshot, encoded_image = load_screenshot_for_state(state, path_to_screenshots)
    prompt: str = prompt__validate_condition(gt_precondition)
    pred_raw_response: str = fetch_openai_vision_completion(prompt, [encoded_image])

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['is_satisfied']
        is_correct: bool = pred_is_met == gt_is_met
    except:
        pred_rationale = None
        pred_is_met = None
        is_correct = False

    return {
        # metadata
        "state_id" : state['data']["id"],
        "action_id" : action['data']["id"],
        "path_to_screenshot": path_to_screenshot,
        # gt
        "gt_is_met" : gt_is_met,
        "gt_precondition": gt_precondition,
        # preds
        "pred_rationale": pred_rationale,
        "pred_is_met" : pred_is_met,
        "pred_raw_response": pred_raw_response,
        # eval
        "is_correct": is_correct,
    }

def validate_precondition(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model_kwargs: Dict[str, str],
    sop: Optional[str] = None,
) -> pd.DataFrame:
    """
        Evaluate pre-condition.
        
        A "TRUE" example is the actual screenshot corresponding to the `gt_precondition`
        A "FALSE" example is a random screenshot from the task trace that occured before the action corresponding to the `gt_precondition`
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_id: int = gt_task_data['webarena']['task_id']

    # group task pipeline into sets of two
    random.seed(1)
    results: List[Dict[str, Any]] = []
    for idx in tqdm(range(0, len(gt_trace) - 1, 2), desc=f"Task: {task_id}"): # -1 because last state has not following action
        state: Dict[str, str] = gt_trace[idx]
        action: Dict[str, str] = gt_trace[idx + 1]

        # Load inputs
        gt_precondition: str = action['gt_labels']["precondition"]
        
        # Eval "TRUE" example
        results.append(helper_precondition(state, action, gt_precondition, True, path_to_screenshots))

        # Eval "FALSE" example
        other_states: List[Dict[str, Any]] = [ x for x in gt_trace if x['type'] == 'state' and x['data']['id'] < state['data']['id'] ]
        if len(other_states) > 0:
            random_state: Dict[str, Any] = random.choice(other_states)
            results.append(helper_precondition(random_state, action, gt_precondition, False, path_to_screenshots))

    df = pd.DataFrame(results)
    return df


def helper_actuation(state: Dict[str, Any], action: Dict[str, Any], state_prime: Dict[str, Any], gt_next_action_suggestion: str, gt_actuation: str, gt_is_met: bool, path_to_screenshots: str) -> Dict[str, str]:
    """Helper fx to eval a single POSITIVE or NEGATIVE example."""
    path_to_screenshot_before, encoded_image_before = load_screenshot_for_state(state, path_to_screenshots)
    path_to_screenshot_after, encoded_image_after = load_screenshot_for_state(state_prime, path_to_screenshots)
    prompt: str = prompt__validate_actuation(gt_next_action_suggestion, gt_actuation)
    pred_raw_response: str = fetch_openai_vision_completion(prompt, [encoded_image_before, encoded_image_after])

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_taken']
        is_correct: bool = pred_is_met == gt_is_met
    except:
        pred_rationale = None
        pred_is_met = None
        is_correct = False

    return {
        # metadata
        "state_id" : state['data']["id"],
        "action_id" : action['data']["id"],
        "path_to_screenshot_before": path_to_screenshot_before,
        "path_to_screenshot_after": path_to_screenshot_after,
        # gt
        "gt_next_action_suggestion": gt_next_action_suggestion,
        "gt_actuation": gt_actuation,
        "gt_is_met" : gt_is_met,
        # preds
        "pred_rationale": pred_rationale,
        "pred_is_met" : pred_is_met,
        "pred_raw_response": pred_raw_response,
        # eval
        "is_correct": is_correct,
    }
    
def validate_actuation(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model_kwargs: Dict[str, str],
    sop: Optional[str] = None,
) -> pd.DataFrame:
    """
        Evaluate actuation.
        
        A "TRUE" example is the actual before/after screenshot from the `gt_action`
        A "FALSE" example is the before and before screenshot (i.e. no change from action)
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_id: int = gt_task_data['webarena']['task_id']

    # group task pipeline into sets of (S, A, S')
    random.seed(1)
    results: List[Dict[str, Any]] = []
    for idx in tqdm(range(0, len(gt_trace) - 2, 2), desc=f"Task: {task_id}"): # -1 because last state has not following action
        state: Dict[str, str] = gt_trace[idx]
        action: Dict[str, str] = gt_trace[idx + 1]
        state_prime: Dict[str, str] = gt_trace[idx + 2]

        # Load inputs
        gt_next_action_suggestion: str = state['gt_labels']["next_action_suggestion"]
        gt_actuation: str = state['gt_labels']["actuation"]
        
        # Eval "TRUE" example
        results.append(helper_actuation(state, action, state_prime, gt_next_action_suggestion, gt_actuation, True, path_to_screenshots))

        # Eval "FALSE" example
        results.append(helper_actuation(state, action, state, gt_next_action_suggestion, gt_actuation, False, path_to_screenshots))

    df = pd.DataFrame(results)
    return df

def build_prompt_s_a_sequence(trace: Dict[str, Any], path_to_screenshots: str) -> Tuple[Dict[str, Any], List[str]]:
    # Loop through trace, interleaving screenshots (states) and actions
    prompt_s_a_sequence: List[str] = []
    paths_to_screenshots: List[str] = []
    for item in trace:
        if item['type'] == 'state':
            path_to_screenshot, encoded_image = load_screenshot_for_state(item, path_to_screenshots)
            paths_to_screenshots.append(path_to_screenshot)
            prompt_s_a_sequence.append({
                "role": "user", 
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
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
    return prompt_s_a_sequence, paths_to_screenshots

def helper_task_completion(gt_trace: Dict[str, Any], task_descrip: str, sop: str, gt_is_met: bool, path_to_screenshots: str, task_type: str) -> Dict[str, str]:
    """Helper fx to eval a single POSITIVE or NEGATIVE example."""
    prompt_s_a_sequence, paths_to_screenshots = build_prompt_s_a_sequence(gt_trace, path_to_screenshots)
    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_completion__intro(task_descrip)
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_completion__close()
        }]
    }
    # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
    pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4-vision-preview', temperature=0.0)

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_is_met: bool = pred_json['was_completed']
        is_correct: bool = pred_is_met == gt_is_met
    except:
        pred_rationale = None
        pred_is_met = None
        is_correct = False

    return {
        # gt
        "gt_is_met" : gt_is_met,
        "paths_to_screenshots" : paths_to_screenshots,
        # preds
        "pred_rationale": pred_rationale,
        "pred_is_met" : pred_is_met,
        "pred_raw_response": pred_raw_response,
        # eval
        "is_correct": is_correct,
        "task_type": task_type,
    }

def validate_task_completion(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model_kwargs: Dict[str, str],
    sop: Optional[str] = None,
    n_negative_samples: int = 3,
) -> pd.DataFrame:
    """
        Evaluate overall task completion success.
        
        If S_1, S_2, ..., S_n are the sequence of screenshots, then...
        
            For "task completed":
                A "TRUE" example is sequence of screenshots of the form [ S_1, ..., S_n ]
                A "FALSE" example is any sequence terminated before S_n, so [S_1, ..., S_j < S_n]
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_id: int = gt_task_data['webarena']['task_id']
    task_descrip: str = gt_task_data['webarena']['intent']
    results: List[Dict[str, Any]] = []

    # Eval "TRUE" example
    results.append(helper_task_completion(gt_trace, task_descrip, sop, True, path_to_screenshots, task_type='true'))

    # Eval "FALSE" examples
    random.seed(1)
    states = [ x for x in gt_trace if x['type'] == 'state' ]
    n_states: int = len(states)
    for i in tqdm(range(n_negative_samples), desc=f'Task: {task_id} negative examples'):
        random_end: int = random.randint(0, n_states - 1)
        gt_trace_negative: List[Dict[str, Any]] = [ x for x in gt_trace if x['data']['id'] < states[random_end]['data']['id'] - 1 ] # -1 to chop off preceding action as well
        results.append(helper_task_completion(gt_trace_negative, task_descrip, sop, False, path_to_screenshots, task_type='truncate'))

    df = pd.DataFrame(results)
    return df


def helper_task_trajectory(gt_trace: Dict[str, Any], task_descrip, sop: str, gt_is_met: bool, path_to_screenshots: str, task_type: str) -> Dict[str, str]:
    prompt_s_a_sequence, paths_to_screenshots = build_prompt_s_a_sequence(gt_trace, path_to_screenshots)
    intro_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_trajectory__intro(task_descrip)
        }]
    }
    close_prompt: Dict[str, str] = {
        "role" : "user",
        "content" : [{
            "type" : "text",
            "text" : prompt__validate_task_trajectory__close(sop)
        }]
    }
    # Feed (S, A, S', A', S'', A'', ...) -- i.e. all screenshots at once
    messages: List[str] = [intro_prompt] + prompt_s_a_sequence + [close_prompt]
    pred_raw_response: str = _fetch_openai_completion(messages, model='gpt-4-vision-preview', temperature=0.0)

    # Evaluate
    try:
        pred_json = json.loads(pred_raw_response.replace("```json", "").replace("```", "").strip())
        pred_rationale: Dict[str, str] = pred_json['rationale']
        pred_inaccurate_steps: List[str] = pred_json.get('inaccurate_steps', [])
        pred_is_met: bool = pred_json['was_accurate'] if 'was_accurate' in pred_json else pred_json['was_acurate']
        is_correct: bool = pred_is_met == gt_is_met
    except:
        pred_rationale = None
        pred_is_met = None
        is_correct = False

    return {
        # gt
        "gt_is_met" : gt_is_met,
        "paths_to_screenshots" : paths_to_screenshots,
        # preds
        "pred_rationale": pred_rationale,
        "pred_inaccurate_steps": pred_inaccurate_steps,
        "pred_is_met" : pred_is_met,
        "pred_raw_response": pred_raw_response,
        # eval
        "is_correct": is_correct,
        "task_type": task_type,
    }


def validate_task_trajectory_valid(
    gt_task_data: Dict[str, Any],
    path_to_screenshots: str,
    model_kwargs: Dict[str, str],
    sop: str,
    n_negative_samples: int = 3,
) -> pd.DataFrame:
    """
        Evaluate overall task completion success.
        
        If S_1, S_2, ..., S_n are the sequence of screenshots, then...
            For "trajectory valid":
                A "TRUE" example is [ S_1, ..., S_n ]
                A "FALSE" example is anything else, e.g. shuffling the order of screenshots, skipping screenshots, etc.
    """
    gt_trace: Dict[str, str] = gt_task_data["trace"]
    task_id: int = gt_task_data['webarena']['task_id']
    task_descrip: str = gt_task_data['webarena']['intent']
    results: List[Dict[str, Any]] = []

    # Eval "TRUE" example
    results.append(helper_task_trajectory(gt_trace, task_descrip, sop, True, path_to_screenshots, task_type='true'))

    # Eval "FALSE" examples
    random.seed(1)
    states = [ x for x in gt_trace if x['type'] == 'state' ]
    n_states: int = len(states)
    ## Skip `skip_len` screenshots at a random interval
    skip_len: str = 2
    for _ in tqdm(range(n_negative_samples), desc=f'Task: {task_id} negative examples (skip)'):
        random_skip_idx: int = random.randint(0, n_states - skip_len)
        skip_state_ids: List[int] = [ states[random_skip_idx + i]['data']['id'] for i in range(skip_len) ]
        skip_action_ids: List[int] = [ x-1 for x in skip_state_ids ]
        gt_trace_negative: List[Dict[str, Any]] = [ x for x in gt_trace if x['data']['id'] not in skip_state_ids  and x['data']['id'] not in skip_action_ids ]
        results.append(helper_task_trajectory(gt_trace_negative, task_descrip, sop, False, path_to_screenshots, task_type='skip'))
    ## Shuffle 2 random screenshots
    for _ in tqdm(range(n_negative_samples), desc=f'Task: {task_id} negative examples (shuffle)'):
        shuffle_id_1: int = states[random.randint(0, max(0, n_states - 2))]['data']['id'] # -2 so we ignore the last state (don't shuffle b/c no subsequent action)
        shuffle_id_2: int = random.choice([ x['data']['id'] for x in states[:-1] if x['data']['id'] != shuffle_id_1 ])
        gt_trace_negative: List[Dict[str, Any]] = []
        for idx, x in enumerate(gt_trace):
            if x['data']['id'] in [ shuffle_id_1+1, shuffle_id_2+1 ]:
                # skip actions after shuffled states
                continue
            elif x['data']['id'] == shuffle_id_1:
                new_state_idx: Dict[str, Any] = [ idx2 for idx2, x2 in enumerate(gt_trace) if x2['data']['id'] == shuffle_id_2][0]
                gt_trace_negative.append(gt_trace[new_state_idx]) # add state
                gt_trace_negative.append(gt_trace[new_state_idx+1]) # add action
            elif x['data']['id'] == shuffle_id_2:
                new_state_idx: Dict[str, Any] = [ idx2 for idx2, x2 in enumerate(gt_trace) if x2['data']['id'] == shuffle_id_1][0]
                gt_trace_negative.append(gt_trace[new_state_idx]) # add state
                gt_trace_negative.append(gt_trace[new_state_idx+1]) # add action
            else:
                gt_trace_negative.append(x)
        assert len(gt_trace) == len(gt_trace_negative), f"Length of `gt_trace` ({len(gt_trace)}) != length of `gt_trace_negative` ({len(gt_trace_negative)})"
        results.append(helper_task_trajectory(gt_trace_negative, task_descrip, sop, False, path_to_screenshots, task_type='shuffle'))
        
    df = pd.DataFrame(results)
    return df

def main(args):
    path_to_task_dir: str = args.path_to_task_dir
    path_to_output_dir: str = args.path_to_output_dir
    is_include_sop: bool = args.is_include_sop
    is_actuation: bool = args.is_actuation
    is_precondition: bool = args.is_precondition
    is_task_completion: bool = args.is_task_completion
    is_task_trajectory_valid: bool = args.is_task_trajectory_valid
    
    assert sum([is_actuation, is_precondition, is_task_completion, is_task_trajectory_valid]) == 1, "Must specify EXACTLY ONE of --is_actuation or --is_precondition or --is_task_completion or --is_task_trajectory_valid"
    
    if is_include_sop:
        raise NotImplementedError("SOP.txt is not yet supported")
    
    # Output dir
    demo_name: str = os.path.basename(path_to_task_dir)
    path_to_output_dir: str = os.path.join(path_to_output_dir, demo_name)
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Find input files
    inputs = load_files_for_task(path_to_task_dir)

    # Execute eval
    output_file_name: str = ""
    if is_actuation:
        df = validate_actuation(
            inputs.gt_task_data, inputs.path_to_screenshots, inputs.model_kwargs, inputs.sop
        )
        output_file_name = "actuation"
    elif is_precondition:
        df = validate_precondition(
            inputs.gt_task_data, inputs.path_to_screenshots, inputs.model_kwargs, inputs.sop
        )
        output_file_name = "precondition"
    elif is_task_completion:
        df = validate_task_completion(
            inputs.gt_task_data, inputs.path_to_screenshots, inputs.model_kwargs, inputs.sop
        )
        output_file_name = "task_completion"
    elif is_task_trajectory_valid:
        df = validate_task_trajectory_valid(
            inputs.gt_task_data, inputs.path_to_screenshots, inputs.model_kwargs, inputs.sop
        )
        output_file_name = "task_trajectory_valid"
    else:
        raise ValueError("Must specify either --is_actuation or --is_precondition or --is_task_completion or --is_task_trajectory_valid")

    # Print metrics
    accuracy: float = df['is_correct'].mean() if 'is_correct' in df.columns else 'N/A'
    all_correct: bool = df['is_correct'].all() if 'is_correct' in df.columns else 'N/A'
    print(f"Task: {path_to_task_dir}")
    print(f"Accuracy: {accuracy}")
    print(f"All correct? {all_correct}")
    df.to_csv(os.path.join(path_to_output_dir, f"validate_{output_file_name}.csv"))

if __name__ == "__main__":
    args, __ = parse_args()
    main(args)