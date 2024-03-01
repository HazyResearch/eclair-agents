import json
import signal
import subprocess
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import datetime

# from eclair.utils.helpers import run_validators


################################################
#
# Log AI agent as completes task
#
################################################

LIST_OF_BROWSER_APPLICATIONS = ["Google Chrome", "Firefox", "Safari"]


class LoggedItemBaseClass:
    def __init__(
        self,
        id: Optional[int] = None,
        step: Optional[int] = None,
        timestamp: Optional[datetime.datetime] = None,
        secs_from_start: Optional[float] = None,
        **kwargs
    ):
        self.id: Optional[int] = id
        self.step: Optional[int] = step
        self.timestamp: Optional[datetime.datetime] = timestamp
        self.secs_from_start: Optional[float] = secs_from_start

    def __repr__(self) -> Dict[str, Any]:
        return self.__dict__

    def __str__(self) -> str:
        return str(self.__dict__)

    def to_json(self) -> Dict[str, Any]:
        return self.__dict__ | {
            "id": self.id,
            "step": self.step,
            "timestamp": self.timestamp.isoformat(),
            "secs_from_start": self.secs_from_start,
        }


class State(LoggedItemBaseClass):
    def __init__(
        self,
        url: Optional[str],
        tab: Optional[str],
        json_state: Optional[List[Dict[str, str]]],
        html: Optional[str],
        screenshot_base64: Optional[str],
        path_to_screenshot: Optional[str],
        window_position: Dict[
            str, int
        ],  # position of active application's focused window
        window_size: Dict[str, int],  # size of active application's focused window
        active_application_name: str,  # name of active application (e.g. "Google Chrome")
        screen_size: Dict[str, str],  # size of entire laptop screen
        is_headless: bool = False,  # if TRUE, then state was taken from browser is running in headless mode
        **kwargs
    ):
        super().__init__(**kwargs)
        self.url: Optional[str] = url
        self.tab: Optional[str] = tab
        self.json_state: Optional[List[Dict[str, str]]] = json_state
        self.html: Optional[str] = html
        self.screenshot_base64: Optional[str] = screenshot_base64
        self.path_to_screenshot: Optional[str] = path_to_screenshot
        self.window_position: Dict[str, int] = window_position  # x, y
        self.window_size: Dict[str, int] = window_size  # width, height
        self.active_application_name: str = active_application_name
        self.screen_size: Dict[
            str, str
        ] = screen_size  # width, height of entire laptop screen
        self.is_headless: bool = is_headless

    def __repr__(self) -> str:
        return f"State(id={self.id}, step={self.step}, url={self.url}, active_application_name={self.active_application_name}, screenshot={self.path_to_screenshot})"
    
    def to_json(self) -> Dict[str, Any]:
        return (
            super().to_json()
            | {
                "screenshot_base64": None,  # don't save screenshot to json b/c messy
            }
            | {
                "json_state": json.dumps(self.json_state)
                if self.json_state
                else None  # Compress `json_state` list into string for readability when printed out to trace.json
            }
        )

    def is_browser(self) -> bool:
        """Return TRUE if state was recorded in a web browser"""
        return self.active_application_name in LIST_OF_BROWSER_APPLICATIONS


class Action(LoggedItemBaseClass):
    def __init__(
        self,
        action: str,
        actuation: str,
        is_valid: bool = False,
        executed_code: Optional[str] = None,
        feedback: Optional[str] = "",
        rationale: Optional[str] = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.action: str = action  # english descrip of action
        self.actuation: str = actuation  # original code to actuate action
        self.is_valid: bool = is_valid  # If TRUE, then action doesn't need to be backtracked
        self.executed_code: Optional[str] = executed_code  # code that was actually executed (might be slightly diff than `actuation`)
        self.rationale: Optional[str] = rationale  # Rationale for why action was selected
        self.feedback: Optional[str] = feedback  # Feedback after having taken action
        self.was_run: bool = False  # If TRUE, then actuation code has been run
        self.element_attributes: Optional[
            str
        ] = None  # If action was taken in browser, then this is the attributes of the element that was interacted with


class Suggestion(LoggedItemBaseClass):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> Dict[str, Any]:
        return f"Suggestion(action={self.action}, is_completed_answer={self.is_completed_answer}, feedback={self.feedback}"
class Validation(LoggedItemBaseClass):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.name: Optional[str] = kwargs.get("name", None) # name of Validator
        self.is_valid: bool = kwargs.get("is_valid", False)
        self.feedback: Optional[str] = kwargs.get("feedback", None)
        self.validations: List["Validation"] = kwargs.get("validations", [])
    
    def __repr__(self) -> Dict[str, Any]:
        return f"Validation(name={self.name}, is_valid={self.is_valid}, feedback={self.feedback}, validations={[ x.name for x in self.validations ]}"

    def to_json(self) -> Dict[str, Any]:
        return super().to_json() | {
            'validations': [ x.to_json() for x in self.validations ],
        }


class TaskLog:
    def __init__(self, task: str):
        self.task: str = task  # string description of task
        self.sop: str = ""  # SOP delineating steps of task
        self.task_ui: str = ""  # name of website/application task occurs in
        self.states: List[State] = []
        self.actions: List[Action] = []
        self.suggestions: Dict[
            str, List[Suggestion]
        ] = {}  # [key] = type of suggestion, [value] = list of suggestions
        self.validations: Dict[
            str, List[Validation]
        ] = {}  # [key] = type of validation, [value] = list of validations
        self.log_id: int = (
            0  # incremented by 1 each time we log something, to preserver ordering
        )
        self.is_completed_success: bool = False  # TRUE if task completed successfully

    @classmethod
    def from_json(self, json: Dict[str, Any]) -> "TaskLog":
        json = json.get("log", {})
        task_log: TaskLog = TaskLog(json.get("task"))
        for action in json.get("actions", []):
            task_log.actions.append(Action(**action))
        for state in json.get("states", []):
            task_log.states.append(State(**state))
        for suggestion in json.get("suggestions", {}):
            task_log.suggestions[suggestion] = [
                Suggestion(**s) for s in json.get("suggestions").get(suggestion)
            ]
        for validation in json.get("validations", {}):
            task_log.validations[validation] = [
                Validation(**v) for v in json.get("validations").get(validation)
            ]
        task_log.log_id = json.get("log_id")
        task_log.is_completed_success = json.get("is_completed_success")
        return task_log

    def to_json(self) -> Dict[str, str]:
        return {
            "task": self.task,
            "states": [x.to_json() for x in self.states],
            "actions": [x.to_json() for x in self.actions],
            "suggestions": {
                key: [val.to_json() for val in vals]
                for key, vals in self.suggestions.items()
            },
            "validations": {
                key: [val.to_json() for val in vals]
                for key, vals in self.validations.items()
            },
            "log_id": self.log_id,
            "is_completed_success": self.is_completed_success,
        }

    def __repr__(self) -> str:
        return f"TaskLog(task={self.task}, n_states=len({len(self.states)}), n_actions={len(self.actions)}"

    def get_sop(self) -> str:
        return self.sop

    def set_sop(self, sop: str) -> None:
        self.sop = sop

    def get_start_state(self) -> State:
        """Return the first state logged."""
        if len(self.states) == 0:
            raise ValueError("No states logged yet, but you called `get_start_state()`")
        return self.states[0]

    def get_current_state(self) -> State:
        """Return the most recent state logged."""
        if len(self.states) == 0:
            raise ValueError(
                "No states logged yet, but you called `get_current_state()`"
            )
        return self.states[-1]

    def get_previous_state(self) -> Optional[State]:
        """Return the state before the most recent state."""
        return self.states[-2] if len(self.states) >= 2 else None

    def get_current_action(self) -> Action:
        """Return the most recent action logged."""
        if len(self.actions) == 0:
            raise ValueError(
                "No actions logged yet, but you called `get_current_action()`"
            )
        return self.actions[-1]

    def get_actions(self) -> List[Action]:
        """Return all actions logged so far."""
        return self.actions

    def get_current_suggestion(self, label: str) -> Optional[str]:
        """Get the most recent suggestion for a given label."""
        if label not in self.suggestions or len(self.suggestions[label]) == 0:
            return None
        return self.suggestions[label][-1]

    def get_current_validation(self, label: str) -> Optional[str]:
        """Get the most recent validation for a given label."""
        if label not in self.validations or len(self.validations[label]) == 0:
            return None
        return self.validations[label][-1]

    def get_current_step(self) -> int:
        if len(self.states) == 0:
            return 0
        return self.get_current_state().step

    def log_state(self, state: Dict[str, Any], step: int):
        self._log_helper("state", state, step)

    def log_action(self, action: str, step: int):
        self._log_helper("action", action, step)

    def log_validation(self, validation: Dict[str, Any], step: int, label: str):
        self._log_helper("validation", validation, step, label)

    def log_suggestion(self, suggestion: Dict[str, Any], step: int, label: str):
        self._log_helper("suggestion", suggestion, step, label)

    def _log_helper(
        self, log_type: str, data: Any, step: int, label: Optional[str] = None
    ):
        # update data (action/state/suggestion/validation/etc.) with metadata
        data.id = self.log_id  # unique autoincrementing integer id for each item logged
        data.step = (
            step  # overall step in task, where each 'step' is a (State, Action) pair
        )
        data.timestamp = datetime.datetime.now()  # timestamp of when item was logged
        data.secs_from_start = (
            (datetime.datetime.now() - self.states[0].timestamp).total_seconds()
            if len(self.states) > 0
            else 0
        )
        
        # validations can have sub-validations, so we need to recursively update their ids
        if log_type == "validation":
            def update_sub_validations(validation: Validation, id: int) -> None:
                for sub_validation in validation.validations:
                    id += 1
                    sub_validation.id = id
                    sub_validation.step = validation.step
                    sub_validation.timestamp = validation.timestamp
                    sub_validation.secs_from_start = validation.secs_from_start
                    id = update_sub_validations(sub_validation, id)
                return id
            self.log_id = update_sub_validations(data, self.log_id)

        if log_type == "state":
            self.states.append(data)
        elif log_type == "action":
            self.actions.append(data)
        elif log_type == "validation":
            if label not in self.validations:
                self.validations[label] = []
            self.validations[label].append(data)
        elif log_type == "suggestion":
            if label not in self.suggestions:
                self.suggestions[label] = []
            self.suggestions[label].append(data)
        else:
            raise ValueError(f"Invalid log_type: {log_type}")

        # increment id for next item logged
        self.log_id += 1

################################################
#
# Collect task demonstrations
#
################################################


class UserAction:
    
    """Store an action taken by the user.

    Possible args:
        self.timestamp: Optional[str] = None # seconds since start
        self.start_timestamp: Optional[str] = None # If action takes a while to complete (e.g. multiple keystrokes), then this is the start time
        self.end_timestamp: Optional[str] = None # If action takes a while to complete (e.g. multiple keystrokes), then this is the end time
        self.type: Optional[str] = None
        self.x: Optional[int] = None
        self.y: Optional[int] = None
        self.button: Optional[str] = None
        self.pressed: Optional[bool] = None
        self.key: Optional[str] = None
        self.element_attributes: Optional[str] = None
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trace:
    """Store a trace of user actions on their computer."""

    def __init__(self):
        self.log: List[Tuple[str, Union[State, UserAction]]] = []
        self.last_state: Optional[State] = None
        self.last_action: Optional[UserAction] = None

    def log_state(self, state: State):
        # Only save webpage stuff if using a browser
        self.log.append(
            {
                "type": "state",
                "data": state,
            }
        )
        self.last_state = state

    def log_action(self, action: UserAction):
        # Only save element_attributes if using browser
        if self.last_state is not None and (
            self.last_state.active_application_name not in LIST_OF_BROWSER_APPLICATIONS
        ):
            if hasattr(action, "element_attributes"):
                del action.element_attributes

        self.log.append(
            {
                "type": "action",
                "data": action,
            }
        )
        self.last_action = action

    def to_json(self) -> List[Dict[str, str]]:
        jsonified: List[Dict[str, str]] = []
        start_timestamp: Optional[datetime.datetime] = None
        for obj in self.log:
            data: Dict = obj.get("data").__dict__

            # Validation
            assert "timestamp" in data, f"Timestamp not found in data: {data}"

            # Add fields
            if start_timestamp is None:
                start_timestamp = data["timestamp"]
            data["secs_from_start"] = (
                (data["timestamp"] - start_timestamp).total_seconds()
                if start_timestamp is not None
                else 0
            )

            # JSONify
            for key, value in data.items():
                if isinstance(value, datetime.datetime):
                    data[key] = value.isoformat()

            jsonified.append(
                {
                    "type": obj.get("type"),
                    "data": data,
                }
            )
        return jsonified


class ScreenRecorder:
    """
    Helper class for screen recording.

    Usage:
    ```
        recorder = ScreenRecorder("test.mp4")
        recorder.start() # starts recording
        recorder.stop() # stops recording, saves recording to file
    """

    def __init__(self, path_to_screen_recording: str) -> None:
        self.path_to_screen_recording: str = path_to_screen_recording

    def start(self) -> None:
        self.proc = subprocess.Popen(
            ["screencapture", "-v", self.path_to_screen_recording],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        os.kill(self.proc.pid, signal.SIGINT)
        # Poll until screen recording is done writing to file
        while self.proc.poll() is None:
            pass


if __name__ == "__main__":
    recorder = ScreenRecorder("test.mp4")
    recorder.start()
    print("Started...")
    time.sleep(2)
    recorder.stop()
    print("Stopped. Saving...")
