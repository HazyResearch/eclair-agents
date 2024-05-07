"""
Usage:

python3 main.py --task "You are a physician in Epic. Please mark all clinical notes for this patient as reviewed." --executor uniagent
"""

import argparse
import datetime
import signal
import os
import json
import time
from typing import Callable, Dict, List, Optional, Tuple
from eclair.utils.logging import ScreenRecorder, TaskLog
from eclair.utils.executors import Environment
from eclair.utils.helpers import (
    log,
    signal_handler,
    is_handler_running_flag,
    get_rel_path,
)
from eclair.utils.constants import (
    TASK_LIST,
    EXECUTORS,
)
from eclair.agent.execute.observer import (
    Observer
)
from eclair.agent.execute.uniagent import (
    UniAgent
)
from eclair.agent.execute.validators import (
    FieldFocusedBeforeTypeValidator,
    FieldEmptyBeforeTypeValidator,
    ActionSuggestionIsDiffThanPrevActionValidator,
    ScrollChangedScreenValidator,
    ClickCoordinatesWithinElementValidator,
)
from eclair.utils.executors import Environment
from eclair.utils.helpers import convert_dsl_to_actions, execute_js_scripts, log, run_code, run_validators
from eclair.utils.logging import Action, TaskLog, State, Suggestion, Validation
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="Textual description of task to execute",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=False,
        help="Task id of task to execute",
    )
    parser.add_argument(
        "--ui",
        type=str,
        required=False,
        help="Name of UI / website running task on (i.e., doordash, gmail etc.)",
    )
    parser.add_argument(
        "--env_type",
        default="selenium",
        type=str,
        required=False,
        choices=["selenium", "playwright", "desktop"],
        help="Environment type. Options: [selenium, playwright, desktop]",
    )
    parser.add_argument(
        "--path_to_sop",
        default=None,
        type=str,
        required=False,
        help="Path to SOP file",
    )

    parser.add_argument(
        "--path_to_prior_task_log_json",
        default=None,
        type=str,
        required=False,
        help="Path to trace.json from which we will intialize the agent's history",
    )
    parser.add_argument(
        "--path_to_output_dir",
        default="./outputs/",
        type=str,
        required=False,
        help="Path to output directory",
    )
    parser.add_argument(
        "--executor",
        type=str,
        required=True,
        choices=EXECUTORS,
        help="Name of eclair executor use",
    )
    parser.add_argument(
        "--trace_name",
        type=str,
        required=False,
        default=None,
        help="Name of trace",
    )
    parser.add_argument(
        "--model",
        default="gpt-4-1106-preview",
        type=str,
        required=False,
        help="Name of OpenAI Model to use",
    )
    parser.add_argument(
        "--max_calls",
        default=20,
        type=int,
        required=False,
        help="Max number of times to call GPT-4",
    )
    parser.add_argument(
        "--max_action_validation_attempts",
        default=4,
        type=int,
        required=False,
        help="Max number of times to call GPT-4 to validate action selection",
    )
    parser.add_argument(
        "--is_disable_screen_recorder",
        default=False,
        action="store_true",
        help="If set, DISABLES screen recording.",
    )

    return parser.parse_known_args()


def execute_task_uniagent(
    model_kwargs: Dict[str, str],
    env: Environment,
    task: str,
    task_ui: str,
    path_to_log_file: str,
    path_to_screenshots_dir: str,
    prompts: Dict[str, Dict[str, str]] = {},
    max_calls: int = 20,
    max_action_validation_attempts: int = 3,
    task_log: Optional[TaskLog] = None,
    sop: str = None,
    **kwargs,
) -> Tuple[bool, List[Dict[str, str]]]:
    # Setup components of AI agent
    if task_log is None:
        task_log: TaskLog = TaskLog(task)
    task_log.task = task
    task_log.task_ui = task_ui
    task_log.env_type = env.env_type

    observer: Observer = Observer(
        env=env,
        path_to_screenshots_dir=path_to_screenshots_dir,
        is_take_screenshots=True,
        is_delete_xpath_from_json_state=False,
        is_save_intermediate_bbox_screenshots=True,
    )
    uniagent: UniAgent = UniAgent(
        model_kwargs=model_kwargs,
        sop=sop,
        **prompts.get('uniagent', {}),
    )

    # Logging     
    logger = lambda msg: log(msg, path_to_log_file)
    observer.set_logger(logger)

    # Observe initial state
    state: State = observer.run()
    task_log.log_state(state, task_log.get_current_step())

    # Run model
    for step in range(task_log.get_current_step(), task_log.get_current_step() + max_calls):
        step_cnt: int = step + 1

        # Choose action
        for i in range(max_action_validation_attempts):
            action_suggestion: Suggestion = uniagent.next_action(task_log)
            task_log.log_suggestion(action_suggestion, step_cnt, label='action_suggestion')
            logger(f"====== ACTION: RESPONSE =======\n```\n{action_suggestion.response}\n```")
            
            # Check if task is completed
            if action_suggestion.is_completed:
                task_log.log_action(action_suggestion, step_cnt)
                task_log.is_completed_success = True
                return task_log
            
            # Validate action suggestion
            if observer.env.env_type == 'desktop':
                # Run general validators that don't rely on HTML attributes
                validation: Validation = run_validators(task_log, [
                    ActionSuggestionIsDiffThanPrevActionValidator(),
                    ScrollChangedScreenValidator(),
                    ClickCoordinatesWithinElementValidator(),
                ])
            else:
                # Run validators that rely on HTML attributes only available for webpage elements (e.g. xpath, focus, empty, etc.)
                validation: Validation = run_validators(task_log, [
                    FieldFocusedBeforeTypeValidator(),
                    FieldEmptyBeforeTypeValidator(),
                    ActionSuggestionIsDiffThanPrevActionValidator(),
                    ScrollChangedScreenValidator(),
                    ClickCoordinatesWithinElementValidator(),
                ])
            task_log.log_validation(validation, step_cnt, label='action_suggestion')

            # Check if action was valid
            if validation.is_valid:
                break
            else:
                logger(f"====== VALIDATION: RESPONSE =======\n```\n{validation.feedback}\n```")
                # Sleep two seconds, reobserve state, and try again
                time.sleep(2)
                # Delete previous state's screenshot
                if os.path.exists(task_log.states[-1].path_to_screenshot):
                    os.remove(task_log.states[-1].path_to_screenshot)
                # Create new state
                state: State = observer.run()
                task_log.states = task_log.states[:-1] # Remove last state so we can update it with current webpage state
                task_log.log_state(state, step_cnt)

        # Register action
        action: Action = Action(
            action=action_suggestion.action,
            actuation=action_suggestion.action,
            rationale=action_suggestion.action_rationale,
        )
        task_log.log_action(action, step_cnt)
        
        # Execute code
        code: str = convert_dsl_to_actions(env, action.actuation)
        try:
            executed_code: str = run_code(env, code, logger)
        except Exception as e:
            print("Error running code")
            print(f"Code:\n```{code}```")
            print(f"Exception: {e}")
            return task_log

        # Log executed code
        task_log.actions[-1].executed_code = executed_code
        logger(f"====== EXECUTED CODE =======\n```\n{executed_code}\n```")
        task_log.get_current_action().was_run = True
        # Log HTML element clicked (if browser)
        # TODO
        # if task_log.get_current_state().is_browser():
        #     element_key: Optional[str] = None
        #     if 'CLICK' in action.action:
        #         element_key = 'lastMouseUp'
        #     elif 'SCROLL' in action.action:
        #         element_key = 'lastScrolled'
        #     elif 'TYPE' in action.action:
        #         element_key = 'lastKeyUp'
        #     elif 'PRESS' in action.action:
        #         element_key = 'lastKeyUp'
        #     else:
        #         logger(f"Couldn't map action `{action.action}` to HTML element")

        time.sleep(2)

        # Rerun JS scripts
        execute_js_scripts(env)
        
        # Observe new (i.e. post-action) state
        state: State = observer.run()
        task_log.log_state(state, step_cnt)
        logger(f"====== NEW STATE =======\n```\n{os.path.basename(state.path_to_screenshot)}\n```")

    return task_log



def main(args: argparse.Namespace, env: Optional[Environment] = None) -> Tuple[TaskLog, str, str]:
    """
    From CLI, only uses `args`.
    If called as function, can pass an existing `Environment`. Defaults to creating a new Selenium driver if no environment is provided.
    """
    task: str = args.task
    task_ui: str = args.ui
    if task in TASK_LIST:
        task = TASK_LIST[task]
    trace_name: str = f"{args.trace_name if args.trace_name is not None else args.task[:20]} @ {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # Paths
    path_to_sop: str = args.path_to_sop
    path_to_prior_task_log_json: str = args.path_to_prior_task_log_json
    path_to_output_dir: str = os.path.abspath(
        os.path.join(args.path_to_output_dir, trace_name)
    )
    path_to_log_file: str = os.path.join(path_to_output_dir, f"{trace_name}.log")
    path_to_screenshots_dir: str = os.path.join(path_to_output_dir, f"screenshots/")
    path_to_screen_recording: str = os.path.join(
        path_to_output_dir, f"{trace_name}.mp4"
    )
    if path_to_prior_task_log_json is not None:
        os.makedirs(path_to_prior_task_log_json, exist_ok=True)
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_screenshots_dir, exist_ok=True)

    log(f"Running task: `{task}`", path_to_log_file)

    # Start screen recorder
    if not args.is_disable_screen_recorder:
        print("Starting screen recorder...")
        screen_recorder = ScreenRecorder(path_to_screen_recording)
        screen_recorder.start()

    # Default to creating a new Selenium driver if none are specified
    if env is None:
        if args.env_type == 'selenium':
            # Attaches to Chrome session running on port 9222
            env = Environment(env_type="selenium")
        elif args.env_type == 'playwright':
            # Launches a new Chrome session
            env = Environment(env_type="playwright")
        elif args.env_type == 'desktop':
            env = Environment(env_type="desktop")
        else:
            raise ValueError(f"Invalid environment type: {args.env_type}")
        env.start()

    # If playwright, start trace collection
    if env.env_type == "playwright":
        env.playwright_context.tracing.start(
            screenshots=True, snapshots=True, sources=True
        )

    # Validators
    with open(get_rel_path(__file__, "utils/event_listeners.js"), "r") as fd:
        js_script: str = fd.read()
    env.execute_script(js_script)

    # Catch Ctrl+C (Kill process) so that we can save logs before quitting
    if not args.is_disable_screen_recorder:
        signal.signal(
            signal.SIGINT,
            lambda x, y: signal_handler(
                x,
                y,
                path_to_output_dir,
                trace_name,
                screen_recorder,
                is_handler_running_flag,
            ),
        )

    # Choose proper executor
    mapping = {
        "uniagent": execute_task_uniagent,
    }
    execute_task: Callable = mapping[args.executor]

    # Set run config
    model_kwargs = {
        "model": args.model,
        "temperature": 0.0,
    }
    prompts: Dict[str, Dict[str, str]] = {
        "uniagent": {
            "intro_prompt": "intro_prompt",
            "action_prompt": "action_prompt",
            "outro_prompt": "outro_prompt",
        },
    }

    # If prior trace is specified, load it
    task_log: Optional[TaskLog] = None
    if path_to_prior_task_log_json is not None:
        if os.path.exists(path_to_prior_task_log_json):
            print(f"Loading prior trace from: `{path_to_prior_task_log_json}`")
            task_log = TaskLog.from_json(
                json.load(open(path_to_prior_task_log_json, "r"))
            )
        else:
            print(
                f"Error - Couldn't load prior trace from: `{path_to_prior_task_log_json}`, so reinitializing TaskLog from scratch"
            )

    # if SOP is specified, load it
    sop: Optional[str] = None
    if path_to_sop is not None and os.path.exists(path_to_sop):
        print(f"Loading SOP from: `{path_to_sop}`")
        sop = open(path_to_sop, "r").read()
    else:
        if path_to_sop is not None:
            print(f"Error - Couldn't load SOP from: `{path_to_sop}`")

    # Execute task
    print(f"==> Writing outputs to: `{path_to_output_dir}`")
    print(f"Starting executor: {args.executor}")
    task_log = execute_task(
        model_kwargs,
        env,
        task,
        task_ui,
        path_to_log_file,
        path_to_screenshots_dir,
        task_log=task_log,
        prompts=prompts,
        max_calls=int(args.max_calls),
        sop=sop,
        task_id=args.task_id,
    )

    # Save task log
    json.dump(
        {"log": task_log.to_json()},
        open(os.path.join(path_to_output_dir, f"{trace_name}.json"), "w"),
        indent=2,
    )

    # Save playwright trace (if applicable)
    if env.env_type == "playwright":
        env.playwright_context.tracing.stop(
            path=os.path.join(path_to_output_dir, f"{trace_name}_playwright_trace.zip")
        )

    # Save screen recording to disk
    print("Stopping screen recorder and saving to disk...")
    if not args.is_disable_screen_recorder:
        screen_recorder.stop()

    print(f"Saved everything to: `{path_to_output_dir}`")
    
    return task_log, trace_name, path_to_output_dir


if __name__ == "__main__":
    args, __ = parse_args()
    main(args)
