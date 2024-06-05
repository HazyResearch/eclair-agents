"""
To use this script with a web browser, please first launch Google Chrome with the following: 

$ alias google-chrome="/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome"
$ google-chrome --remote-debugging-port=9222 
$ python3 record.py --name "my_task"

Then, navigate to the website for which you want to trace actions for. 
This traces keystrokes + clicks.
"""
import json
import datetime
from functools import partial
import os
from pynput import mouse, keyboard
import argparse
import multiprocessing
from eclair.utils.executors import Environment
from eclair.utils.helpers import (
    merge_consecutive_keystrokes,
    merge_consecutive_scrolls,
    merge_consecutive_states,
    remove_action_type,
    remove_esc_key,
    execute_js_scripts,
    extract_screenshots_for_demo,
    get_last_element_attributes,
    get_rel_path
)
from typing import Dict, List, Any, Optional

from eclair.utils.logging import (
    LIST_OF_BROWSER_APPLICATIONS,
    State,
    UserAction,
    Trace,
    ScreenRecorder,
)
from eclair.executors.modules.observer.observer import Observer

TRACE_END_KEY = keyboard.Key.esc  # Press this key to terminate the trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--path_to_output_dir", default="./outputs/", type=str, required=False, help="Path to output directory", )
    parser.add_argument( "--is_webarena", action="store_true", help="Set to TRUE if task is webarena task", )
    parser.add_argument( "--is_desktop_only", action="store_true", help="Set to TRUE if we're only going to be using a desktop application (e.g. Epic, Word, Excel, etc.)", )
    parser.add_argument( "--is_record_audio", action="store_true", help="Set to TRUE to record audio from computer's microphone (to enable voice over narration)")
    parser.add_argument( "--is_show_clicks", action="store_true", help="Set to TRUE to show clicks in .mp4")
    parser.add_argument("--valid_application_name", type=str, default=None, help="If specified, only permit this application to be kept in the trace")
    parser.add_argument( "--name", type=str, required=False, help="Name of task being demonstrated" )
    return parser.parse_args()


def print_(*args):
    """Hacky fix needed to get printed statements to left-align in terminal (prob caused by calling `screencapture` command)"""
    print(*args, "\r")


def is_string_in_integer_range(s: str, min_value: int, max_value: int):
    """Assert that the string 's' represents an integer within the specified range [min_value, max_value]."""
    try:
        value = int(s)
        return min_value <= value <= max_value
    except ValueError:
        return False

####################################################
####################################################
#
# Mouse/Keyboard listeners
#
####################################################
####################################################


def on_action_worker(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    path_to_screenshots_dir: str,
    is_desktop_only: bool,
):
    env: Environment = Environment("selenium" if not is_desktop_only else "desktop")
    env.start()
    print_("\r>>>>>>>> GOOD TO START RECORDING WORKFLOW <<<<<<<<<<")
    observer: Observer = Observer(
        env=env,
        path_to_screenshots_dir=path_to_screenshots_dir,
        is_take_screenshots=False,
        is_delete_xpath_from_json_state=False,
        is_save_intermediate_bbox_screenshots=False,
    )
    while True:
        data: Dict[str, Any] = task_queue.get()  # This will block if the queue is empty
        if data is None:
            result_queue.put(None)
            break

        # Get state
        state: State = observer.run()
        state.timestamp = data["timestamp"]
        result_queue.put(state)

        # Get action
        is_application_browser: bool = (
            state.active_application_name in LIST_OF_BROWSER_APPLICATIONS
        )
        action: Dict[str, Any] = {
            key: val
            for key, val in data.items()
            if not (
                key == "element_attributes" and not is_application_browser
            )  # Don't save element_attributes if not using browser
        }
        result_queue.put(UserAction(**action))

        print_({key: val for key, val in action.items() if key != "element_attributes"})

        # Re-execute scripts if webpage has changed
        if is_application_browser and not env.execute_script(
            "return window.isEventListenerLoaded"
        ):
            execute_js_scripts(env)
    env.stop()
    exit()


def on_scroll(
    task_queue: multiprocessing.Queue,
    env: Environment,
    x: int,
    y: int,
    dx: int,
    dy: int,
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "scroll",
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "element_attributes": get_last_element_attributes(env, "lastScrolled"),
        }
    )


def on_click(
    task_queue: multiprocessing.Queue,
    env: Environment,
    x: int,
    y: int,
    button: mouse.Button,
    pressed: bool,
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "mousedown" if pressed else "mouseup",
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "is_right_click": button == mouse.Button.right,
            "pressed": pressed,
            "element_attributes": get_last_element_attributes(
                env, "lastMouseDown" if pressed else "lastMouseUp"
            ),
        }
    )


def on_key_press(
    task_queue: multiprocessing.Queue, env: Environment, key: keyboard.Key
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "keypress",
            "timestamp": timestamp,
            "key": str(key),
            "element_attributes": get_last_element_attributes(env, "lastKeyDown"),
        }
    )
    # Quit if ESC is pressed
    if key == TRACE_END_KEY:
        mouse_listener.stop()
        keyboard_listener.stop()


def on_key_release(
    task_queue: multiprocessing.Queue, env: Environment, key: keyboard.Key
):
    timestamp: datetime.datetime = datetime.datetime.now()
    task_queue.put(
        {
            "type": "keyrelease",
            "timestamp": timestamp,
            "key": str(key),
            "element_attributes": get_last_element_attributes(env, "lastKeyUp"),
        }
    )


if __name__ == "__main__":
    args = parse_args()
    args.name = args.name if args.name is not None else "trace"
    trace_name: str = (
        f"{args.name} @ {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    is_desktop_only: bool = args.is_desktop_only
    is_record_audio: bool = args.is_record_audio
    is_show_clicks: bool = args.is_show_clicks
    valid_application_name: Optional[str] = args.valid_application_name
    path_to_output_dir: str = os.path.join(args.path_to_output_dir, trace_name)
    path_to_screenshots_dir: str = os.path.join(path_to_output_dir, f"screenshots/")
    path_to_webarena_tasks: str = get_rel_path(__file__, "../../eval/webarena/tasks/")
    path_to_screen_recording: str = os.path.join(
        path_to_output_dir, f"{trace_name}.mp4"
    )

    # make dirs
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_screenshots_dir, exist_ok=True)

    # Setup webarena tasks
    webarena_task: Optional[str] = None
    webarena_start_url: Optional[str] = None
    if args.is_webarena:
        assert is_string_in_integer_range(
            args.name, 0, 811
        ), f"Invalid task name for WebArena: `{args.name}`"
        for filename in os.listdir(path_to_webarena_tasks):
            if not filename.endswith(".json") or filename.startswith("test"):
                # Skip non-JSON files and test files
                continue
            task_id: int = int(filename.split(".")[0])
            if int(args.name) == task_id:
                task = json.load(
                    open(os.path.join(path_to_webarena_tasks, filename), "r")
                )
                webarena_task = task["intent"]
                webarena_start_url: str = task["start_url"]
        print_(f"Task: `{webarena_task}`")

    # make dirs
    os.makedirs(path_to_output_dir, exist_ok=True)
    os.makedirs(path_to_screenshots_dir, exist_ok=True)

    # Attach to Chrome session running on port 9222 (if using browser)
    env: Environment = Environment("selenium" if not is_desktop_only else "desktop")
    env.start()
    if args.is_webarena:
        env.get(webarena_start_url)

    # Start Javascript scripts
    execute_js_scripts(env)

    # Start screen recorder
    print_(f"Starting screen recorder{' with audio' if is_record_audio else ''}{' with clicks shown' if is_show_clicks else ''}...")
    screen_recorder = ScreenRecorder(path_to_screen_recording, is_record_audio=is_record_audio, is_show_clicks=is_show_clicks)
    screen_recorder.start()

    # Queues for multiprocessing
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Log initial state
    observer: Observer = Observer(
        env=env,
        path_to_screenshots_dir=path_to_screenshots_dir,
        is_take_screenshots=False,
        is_delete_xpath_from_json_state=False,
        is_save_intermediate_bbox_screenshots=False,
    )
    initial_state: State = observer.run()
    initial_state.timestamp = datetime.datetime.now()
    result_queue.put(initial_state)

    # Start listeners for mouse/keyboard interactions
    action_logger_process = multiprocessing.Process(
        target=on_action_worker,
        args=(task_queue, result_queue, path_to_screenshots_dir, is_desktop_only),
    )
    action_logger_process.start()
    with mouse.Listener(
        on_click=partial(on_click, task_queue, env),
        on_scroll=partial(on_scroll, task_queue, env),
    ) as mouse_listener:
        with keyboard.Listener(
            on_press=partial(on_key_press, task_queue, env),
            on_release=partial(on_key_release, task_queue, env),
        ) as keyboard_listener:
            keyboard_listener.join()
            mouse_listener.join()

    # Save trace
    task_queue.put(None)
    trace: Trace = Trace()
    while True:
        result = result_queue.get()
        if result is None:
            break
        if isinstance(result, UserAction):
            trace.log_action(result)
        elif isinstance(result, State):
            trace.log_state(result)
        else:
            raise ValueError(f"Unknown result type: {type(result)}")

    print_("Done with tracing. Savings results...")
    print_("# of events:", len(trace.log))

    # Stop screen recording and save to disk
    print_("Stopping screen recorder and saving to disk...")
    screen_recorder.stop()

    # Close processes
    action_logger_process.join()
    action_logger_process.close()

    # Get trace
    trace_json: List[Dict[str, Any]] = trace.to_json()
    
    # Save raw trace
    json.dump(
        {"trace": trace_json},
        open(os.path.join(path_to_output_dir, f"[raw] {trace_name}.json"), "w"),
        indent=2,
    )

    # Post processing
    # Merge consecutive scroll events
    trace_json = merge_consecutive_scrolls(trace_json)
    # Remove ESC keypresses
    trace_json = remove_esc_key(trace_json)
    # Remove keyrelease
    trace_json = remove_action_type(trace_json, "keyrelease")
    # Merge consecutive keystrokes in same input field
    trace_json = merge_consecutive_keystrokes(trace_json)
    # Merge consecutive states without intermediate actions (only keep first + last)
    trace_json = merge_consecutive_states(trace_json)
    # Reset data id's
    for i, x in enumerate(trace_json):
        x["data"]["id"] = i
        if "step" in x['data']:
            del x['data']['step']

    # Save trace to disk
    json.dump(
        {"trace": trace_json},
        open(os.path.join(path_to_output_dir, f"{trace_name}.json"), "w"),
        indent=2,
    )
    
    # Resample screenshots
    extract_screenshots_for_demo(path_to_output_dir)

    # Pull out screenshots from screen recording
    print_("Creating screenshots from screen recording...")
    extract_screenshots_for_demo(path_to_output_dir)

    print_(f"DONE. Saved to: `{path_to_output_dir}`")
    os.system("stty sane") # fix terminal line spacing
