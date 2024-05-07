from dotenv import load_dotenv

load_dotenv()
import argparse
import anthropic
import base64
from dataclasses import dataclass
import json
from multiprocessing import Pool
import os
import pickle
import random
import sys
import time
import openai
import google.generativeai as genai
import traceback
from typing import Any, Dict, List, Optional, Tuple
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import json
import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from PIL import Image
from io import BytesIO
from google.api_core.exceptions import InvalidArgument
from typing import Callable, Dict, List
import collections
import os
from together import Together


SYSTEM_PROMPT: str = "You are a helpful assistant that automates digital workflows."
ACTION_TRANSITION: str = f"Action: PRESS(Cmd+Tab)"


class Task:
    """Parse Gdrive folder + .xlsx file into Task object."""

    action_char_map: Dict[str, str] = {
        "mouseup": "C",  # click
        "keystroke": "T",  # type
        "keypress": "K",
        "scroll": "S",
    }

    def __init__(self):
        self.gdrive_folder: Dict[str, str] = None
        # Labels from .xlsx file
        self.task_id: Optional[int] = None
        self.person: Optional[str] = None
        self.possible: Optional[str] = None
        self.difficulty: Optional[str] = None
        self.notes: Optional[str] = None
        self.site: Optional[str] = None
        self.task: Optional[str] = None
        self.intent_template: Optional[str] = None
        self.is_na: Optional[bool] = None
        self.gdrive_link: Optional[str] = None
        # Lazy load
        self.sop: Optional[str] = None  # contents of SOP.txt
        self.trace: Optional[Dict[str, str]] = (
            None  # contents of trace.json as parsed JSON dict
        )
        self.path_to_video: Optional[str] = None  # path to .mp4
        self.screenshots: Optional[List[str]] = None  # list of screenshot URLs
        self.trace_actions: Optional[str] = None  # trace action string, e.g. "CCSSSTT"
        self.recording_length: Optional[str] = None  # length of recording in seconds

    def get_sop(self, drive) -> Optional[str]:
        if not hasattr(self, "sop") or self.sop is None:
            # Fetch SOP.txt
            for file in get_files_in_folder(drive, self.gdrive_folder["id"]):
                if file["title"].startswith("SOP"):
                    self.sop = file.GetContentString(mimetype="text/plain")
                    break
        return self.sop

    def get_trace(self, drive) -> Optional[str]:
        if not hasattr(self, "trace") or self.trace is None:
            # Fetch trace.json
            for file in get_files_in_folder(drive, self.gdrive_folder["id"]):
                if file["title"].endswith(".json") and not file["title"].startswith(
                    "[raw]"
                ):
                    self.trace = json.loads(
                        file.GetContentString(mimetype="text/plain")
                    )
                    break
        return self.trace

    def get_path_to_video(self, drive) -> Optional[str]:
        if not hasattr(self, "path_to_video") or self.path_to_video is None:
            os.makedirs("/tmp/videos/", exist_ok=True)
            # Fetch recording.mp4
            for file in get_files_in_folder(drive, self.gdrive_folder["id"]):
                if file["title"].endswith(".mp4"):
                    random_id: str = str(random.choice(range(1000000)))
                    self.path_to_video = f"/tmp/videos/{random_id}.mp4"
                    file.GetContentFile(self.path_to_video, mimetype="video/mp4")
        return self.path_to_video

    def is_video_exists(self, drive) -> bool:
        for file in get_files_in_folder(drive, self.gdrive_folder["id"]):
            if file["title"].endswith(".mp4"):
                return True
        return False

    def is_screenshots_exists(self, drive) -> bool:
        for folder in get_folders_in_folder(drive, self.gdrive_folder["id"]):
            if folder["title"].startswith("screenshots"):
                return True
        return False

    def get_trace_action_str(self, drive) -> Optional[str]:
        if self.trace_actions is None:
            trace = self.get_trace(drive)
            self.trace_actions = ""
            for trace_item in trace["trace"]:
                if trace_item["type"] == "action":
                    self.trace_actions += self.action_char_map[
                        trace_item["data"]["type"]
                    ]
        return self.trace_actions

    def get_recording_length(self, drive) -> Optional[str]:
        if self.recording_length is None:
            # Fetch recording.mp4
            assert self.path_to_video is not None, "Path to video is not set"
            with VideoFileClip(self.path_to_video) as video:
                self.recording_length = video.duration
        return self.recording_length


@dataclass
class QACheck:
    """Class to store QA check result."""

    person: str  # Full name of person who did task
    task_id: int  # Task ID
    folder_url: str  # URL to Google Drive folder
    is_valid: bool  # TRUE if task passes, FALSE if need to redo this task
    note: Optional[str] = None  # Any additional info to note
    fixed_sop: Optional[str] = (
        None  # New SOP to replace old SOP, if `is_valid=False` and SOP is fixable
    )
    fixed_trace_json: Optional[str] = (
        None  # New trace.json to replace old trace.json, if `is_valid=False` and trace.json is fixable
    )
    fixed_path_to_video: Optional[str] = (
        None  # New .mp4 replace old .mp4, if `is_valid=False` and .mp4 is fixable
    )
    other: Optional[Dict] = None  # Any other info to store


# Scopes required for reading Google Drive
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
ROOT_FOLDER_ID: str = "1USiWqa4Iak5a-vexVeGYXq-v3rW2zFqE"  # Taken from URL
ARCHIVE_FOLDER_ID: str = "1qNcz5y_DudZHGwBsPTySL5Qyp19j_6Uf"  # Taken from URL


def create_merged_trace(
    path_to_demo_folders: List[str],
    is_interleave: bool,
    is_concatenate: bool,
    is_keep_act: bool = True,
    is_keep_kfs: bool = True,
    random_seed: int = 1,
) -> List[Dict[str, Any]]:
    # Load demos
    logs: Dict[List[Dict[str, Any]]] = collections.defaultdict(list)
    for demo in path_to_demo_folders:
        # Load files
        path_to_trace: str = get_path_to_trace_json(demo)
        path_to_screenshots_dir: str = get_path_to_screenshots_dir(demo)
        # Read files
        trace_json: Dict[str, Any] = json.load(open(path_to_trace, "r"))
        task_id: int = int(trace_json["webarena"]["task_id"])
        prompt_s_a_sequence, paths_to_screenshots = build_prompt_s_a_sequence(
            trace_json["trace"], path_to_screenshots_dir
        )
        for item_idx, item in enumerate(prompt_s_a_sequence):
            logs[task_id].append(
                {
                    "task_id": task_id,
                    "item_idx": item_idx,
                    "item": {"role": "user", "content": item["content"]},
                    "item_type": (
                        "action" if item["content"][0]["type"] == "text" else "state"
                    ),
                }
            )
        # Add a final "transition" action between non-final demos
        if demo != path_to_demo_folders[-1] and len(path_to_demo_folders) > 1:
            logs[task_id].append(
                {
                    "task_id": task_id,
                    "item_idx": item_idx + 1,
                    "item": {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": ACTION_TRANSITION,
                            }
                        ],
                    },
                    "item_type": "action",
                }
            )

    # Create merged log
    random.seed(random_seed)
    merged_log: List[Dict[str, Any]] = []
    if is_concatenate:
        # Concatenate consecutively
        keys: List[int] = sorted(list(logs.keys()))
        random.shuffle(keys)
        for key in keys:
            merged_log.extend(logs[key])
    elif is_interleave:
        # Interleave
        curr_idxs: Dict[int, int] = {task_id: 0 for task_id in logs.keys()}
        non_empty_tasks: List[int] = [
            task_id for task_id, log in logs.items() if len(log) > 0
        ]
        while len(non_empty_tasks) > 0:
            # Randomly choose a task to add
            next_task_id: int = random.choice(non_empty_tasks)
            # Add next state + action from that task into merged log
            merged_log.append(logs[next_task_id][curr_idxs[next_task_id]])
            merged_log.append(logs[next_task_id][curr_idxs[next_task_id + 1]])
            # Move ptr to next item in that task
            curr_idxs[next_task_id] += 2
            # Remove task from non_empty_tasks if it's empty
            if curr_idxs[next_task_id] >= len(logs[next_task_id]):
                non_empty_tasks.remove(next_task_id)

    # Filtering
    if not is_keep_act:
        # Remove actions
        merged_log = [x for x in merged_log if x["item_type"] != "action"]
    if not is_keep_kfs:
        # Remove key frames
        merged_log = [x for x in merged_log if x["item_type"] != "state"]

    # Add UUID to each item
    for idx, item in enumerate(merged_log):
        item["uuid"] = idx

    return merged_log


def get_folders_in_folder(drive, folder_id: str) -> List:
    """Given a Google Drive folder ID, return a list of all subfolders"""
    results = []
    file_list = drive.ListFile(
        {"q": "'%s' in parents and trashed=false" % folder_id}
    ).GetList()
    for f in file_list:
        if f["mimeType"] == "application/vnd.google-apps.folder":  # if folder
            results.append(f)
    return results


def get_files_in_folder(drive, folder_id: str) -> List:
    """Given a Google Drive folder ID, return a list of all files in that folder"""
    results = []
    file_list = drive.ListFile(
        {"q": "'%s' in parents and trashed=false" % folder_id}
    ).GetList()
    for f in file_list:
        if f["mimeType"] != "application/vnd.google-apps.folder":  # if file
            results.append(f)
    return results


def group_tasks_by_id(tasks: List[Task]) -> Dict[str, List[Task]]:
    """Group tasks performed by different people into task_id."""
    grouped_tasks = {}
    for task in tasks:
        if task.task_id not in grouped_tasks:
            grouped_tasks[task.task_id] = []
        grouped_tasks[task.task_id].append(task)

    return grouped_tasks


def get_webarena_task_json(task_id: int) -> Optional[Dict[str, str]]:
    """Given the integer task ID, return the task JSON from the WebArena dataset"""
    path_to_webarena_tasks: str = get_rel_path(__file__, "./tasks/")
    for filename in os.listdir(path_to_webarena_tasks):
        if not filename.endswith(".json") or filename.startswith("test"):
            # Skip non-JSON files and test files
            continue
        file_task_id: int = int(filename.split(".")[0])
        if int(task_id) == file_task_id:
            task = json.load(open(os.path.join(path_to_webarena_tasks, filename), "r"))
            return task
    return None


def _fetch_geminipro_completion(messages: List[Any], model_name: str) -> str:
    """Helper function to call Google's GeminiPro API. Handles rate limit errors and other exceptions"""

    assert model_name in [
        "gemini-1.0-pro",
        "gemini-pro-vision",
    ], f"Unknown model: {model_name}, must be one of 'gemini-1.0-pro', 'gemini-pro-vision'"

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name)

    IMAGE_LIMIT: int = 16 if model_name == "gemini-pro-vision" else 9999999
    img_counter: int = 0

    # Flatten into one user message
    content: List[str] = []
    for idx, message in enumerate(messages):
        for content_obj in message["content"]:
            if content_obj["type"] == "text":
                content.append(content_obj["text"])
            elif content_obj["type"] == "image_url":
                if img_counter >= IMAGE_LIMIT:
                    # Skip image if limit reached
                    continue
                # First 22 characters are "data:image/png;base64,"
                img: Image = Image.open(
                    BytesIO(base64.b64decode(content_obj["image_url"]["url"][22:]))
                )
                content.append(img)
                img_counter += 1
            else:
                raise ValueError(f"Unknown content type: {content_obj['type']}")
    new_messages = [
        {
            "role": "user",
            "parts": content,
        }
    ]

    try:
        response = model.generate_content(new_messages)
    except Exception as e:
        if "Quota exceeded for quota metric" in str(
            e
        ) or "Resource has been exhausted" in str(e):
            print(f"Rate limit exceeded -- waiting 1 min before retrying")
            time.sleep(60)
            return _fetch_geminipro_completion(messages, model_name)
        traceback.print_exc()
        print(f"Unknown error: {e}")
    return response.text


def _fetch_together_completion(messages: List[Any], model_name: str) -> str:

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response.choices[0].message.content


def _fetch_claude_completion(messages: List[Any], model_name: str) -> str:
    """Helper function to call Anthropic's Claude API. Handles rate limit errors and other exceptions"""
    assert model_name in [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ], f"Unknown model: {model_name}, must be one of 'gemini-pro', 'gemini-pro-vision'"

    IMAGE_LIMIT: int = 20  # Max number of images allowed in prompt
    img_counter: int = 0

    for idx, message in enumerate(messages):
        content: List[Dict[str, Any]] = []
        for content_obj in message["content"]:
            if content_obj["type"] == "image_url":
                # Reformulate image content
                if img_counter >= IMAGE_LIMIT:
                    # Skip image if limit reached
                    continue
                image_data: str = content_obj["image_url"]["url"][
                    22:
                ]  # First 22 characters are "data:image/png;base64,"
                content_obj = {
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    }
                }
                img_counter += 1
            content.append(content_obj)
        messages[idx]["content"] = content

    try:
        response = anthropic.Anthropic().messages.create(
            model=model_name,
            max_tokens=4096,
            messages=messages,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Unknown error: {e}")
    return response.content[0].text


def _fetch_openai_completion(messages: List[Any], **kwargs) -> str:
    """Helper function to call OpenAI's Vision API. Handles rate limit errors and other exceptions"""
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            max_tokens=4096,
            **kwargs,
        )
    except openai.RateLimitError:
        print("Rate limit exceeded -- waiting 1 min before retrying")
        time.sleep(60)
        return _fetch_openai_completion(messages, **kwargs)
    except openai.APIError as e:
        traceback.print_exc()
        print(f"OpenAI API error: {e}")
        raise e
    except Exception as e:
        traceback.print_exc()
        print(f"Unknown error: {e}")
        raise e
    return response.choices[0].message.content


def _fetch_completion(
    messages: List[Any], model: str, model_name: Optional[str] = None, **kwargs
) -> str:
    """Master router."""
    is_image: bool = any(
        content["type"] == "image_url"
        for message in messages
        for content in message["content"]
    )
    if model == "GPT4":
        if not model_name:
            model_name: str = (
                "gpt-4-0125-preview" if not is_image else "gpt-4-vision-preview"
            )
        response: str = _fetch_openai_completion(
            messages, model=model_name, temperature=0.0
        )
    elif model == "GeminiPro":
        if not model_name:
            model_name: str = "gemini-1.0-pro" if not is_image else "gemini-pro-vision"
        response: str = _fetch_geminipro_completion(messages, model_name=model_name)
    elif model == "Claude3":
        if not model_name:
            model_name = "claude-3-haiku-20240307"
        response: str = _fetch_claude_completion(messages, model_name=model_name)
    elif model == "Together":
        if not model_name:
            model_name = "mistralai/Mixtral-8x7B-v0.1"
        response: str = _fetch_together_completion(messages, model_name=model_name)
    else:
        raise ValueError(f"Unknown model: {model}")
    return response


def fetch_openai_json_completion(
    prompt: str, system_prompt: Optional[str] = None, **kwargs
) -> str:
    """Helper function to call OpenAI's JSON Completion API. Handles rate limit errors and other exceptions"""
    assert (
        "json" in prompt.lower()
    ), "Error - when using OpenAI JSON endpoint, your prompt must contain the word 'json' but currently doesn't"
    messages = [
        {
            "role": "system",
            "content": system_prompt if system_prompt is not None else SYSTEM_PROMPT,
        },
        {"role": "user", "content": prompt},
    ]
    return _fetch_openai_completion(
        messages,
        response_format={"type": "json_object"},
        **kwargs,
    )


def convert_trace_action_to_dsl(event: Dict[str, str]) -> Dict[str, str]:
    """Converts a trace.json action into DSL format."""
    action_data: Dict[str, str] = {}
    tag_2_type = {
        "input": "text field",
        "button": "button",
        "a": "link",
        "select": "dropdown",
        None: "None",
    }
    if (
        "element_attributes" in event["data"]
        and event["data"]["element_attributes"] is not None
    ):
        if "element" in event["data"]["element_attributes"]:
            attrs: Dict[str, Any] = event["data"]["element_attributes"]["element"]
        else:
            attrs: Dict[str, Any] = event["data"]["element_attributes"]
        elem_x: int = int(attrs["x"]) if "x" in attrs else None
        elem_y: int = int(attrs["y"]) if "y" in attrs else None
        elem_height: int = int(attrs["height"]) if "height" in attrs else None
        elem_width: int = int(attrs["width"]) if "width" in attrs else None
        elem_text: str = attrs.get("text")
        elem_tag: str = attrs.get("tag")
        elem_label: str = attrs.get("label")
        elem_role: str = attrs.get("role")
        elem_placeholder: str = attrs.get("placeholder")
        elem_textified: str = (
            elem_label
            if elem_label
            else (
                elem_role
                if elem_role
                else (elem_text if elem_text else elem_placeholder)
            )
        )
        elem_type = tag_2_type[elem_tag] if elem_tag in tag_2_type else "text"
        elem_xpath: str = attrs.get("xpath")
    else:
        elem_x = None
        elem_y = None
        elem_height = None
        elem_width = None
        elem_text = None
        elem_tag = None
        elem_label = None
        elem_textified = None
        elem_type = None
        elem_xpath = None

    # if event["data"]["type"] in ["mouseup", "keystroke", "keypress"]:
    #     if event["data"]["type"] == "mouseup":
    #         action_data["action"] = (
    #             f"Click on the {elem_type} labeled '{elem_textified}'"
    #         )
    #         action_data["actuation_suggestion"] = {
    #             "action": f"CLICK({int(event['data']['x'])}, {int(event['data']['y'])})",
    #             "element": f"{{'x': {elem_x}, 'y': {elem_y}, 'height': {elem_height}, 'width': {elem_width}, 'text': '{elem_text}', 'tag': '{elem_tag}', 'xpath' : '{elem_xpath}' }}",
    #         }
    #     elif event["data"]["type"] in ["keystroke"]:
    #         keystroke: str = "".join(
    #             [x.replace("'", "") for x in event["data"]["key"].split(" ")]
    #         )
    #         keystroke = keystroke.replace("Key.space", " ")
    #         keystroke = keystroke.replace("Key.shift_r", "")
    #         keystroke = keystroke.replace("Key.shift", "")
    #         action_data["action"] = (
    #             f"Type the string '{keystroke}' in the {elem_type} labeled '{elem_textified}'"
    #         )
    #         action_data["actuation_suggestion"] = {
    #             "action": f'TYPE("{keystroke}")',
    #             "element": f"{{'x': {elem_x}, 'y': {elem_y}, 'height': {elem_height}, 'width': {elem_width}, 'text': '{elem_text}', 'tag': '{elem_tag}', 'xpath' : '{elem_xpath}'}}",
    #         }
    #     elif event["data"]["type"] in ["keypress"]:
    #         keystroke: str = "".join(
    #             [x.replace("'", "") for x in event["data"]["key"].split(" ")]
    #         )
    #         keystroke = keystroke.replace("Key.space", " ")
    #         keystroke = keystroke.replace("Key.", "")
    #         if keystroke.lower() in ["enter", "return"]:
    #             keystroke = "Enter"
    #         action_data["action"] = (
    #             f"Press the key '{keystroke}' in the {elem_type} labeled '{elem_textified}'"
    #         )
    #         action_data["actuation_suggestion"] = {
    #             "action": f'PRESS("{keystroke}")',
    #             "element": f"{{'x': {elem_x}, 'y': {elem_y}, 'height': {elem_height}, 'width': {elem_width}, 'text': '{elem_text}', 'tag': '{elem_tag}', 'xpath' : '{elem_xpath}'}}",
    #         }
    #     elif event["data"]["type"] == "scroll":
    #         action_data["action"] = ""
    #         if event["data"]["dy"] != 0:
    #             direction = "up" if event["data"]["dy"] < 0 else "down"
    #             pixels: int = abs(event["data"]["dy"])
    #             action_data["action"] += f"Scroll {direction} by {pixels} pixels."
    #         if event["data"]["dx"] != 0:
    #             direction = "left" if event["data"]["dx"] < 0 else "right"
    #             pixels: int = abs(event["data"]["dx"])
    #             action_data["action"] += (
    #                 " " if event["data"]["dy"] != 0 else ""
    #             ) + f"Scroll {direction} by {pixels} pixels."
    #         action_data["actuation_suggestion"] = {
    #             "action": f'SCROLL({event["data"]["dx"]},{event["data"]["dy"]})',
    #             "element": "None",
    #         }
    action_data = {}
    action_data["action"] = event["action"]
    action_data["actuation_suggestion"] = event["action"]
    return action_data


def encode_image(path_to_img: str):
    """Base64 encode an image"""
    with open(path_to_img, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_screenshot_for_state(state: str, path_to_screenshots: str) -> Tuple[str, str]:
    path_to_screenshot: str = state["path_to_screenshot"]
    path_to_screenshot: str = path_to_screenshot.split("/")[-1]
    encoded_image: str = encode_image(
        os.path.join(path_to_screenshots, path_to_screenshot)
    )
    return path_to_screenshot, encoded_image


def get_rel_path(file: str, rel_path: str) -> str:
    """Transforms a relative path from a specific file in the package `eclair/src/eclair/ into an absolute path"""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(file)), rel_path)
    )


def load_tasks_excel(
    excel_path="./data/Process Mining Task Demonstrations.xlsx",
) -> pd.DataFrame:
    """Load Excel file containing all tasks."""
    path_to_xlsx: str = get_rel_path(__file__, excel_path)
    dfs: Dict[str, pd.DataFrame] = pd.read_excel(path_to_xlsx, sheet_name=None)
    # Remove sheets that are not tasks
    dfs = {key: df for key, df in dfs.items() if key.startswith("Sheet")}
    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    df = df[
        df["Task ID"].notna() & df["Person"].notna()
    ]  # remove rows with no task ID or person
    return df


def load_demo_folders_from_drive(
    path_to_client_secrets_json: str,
) -> List[Dict[str, str]]:
    # Connect to Google Drive
    drive = init_drive(path_to_client_secrets_json)
    results: List[str] = []
    task_folders = get_folders_in_folder(drive, ROOT_FOLDER_ID)
    for subfolder in tqdm(task_folders, desc="Loading tasks from Gdrive..."):
        if " @ " not in subfolder["title"]:
            print(
                f"WARNING | Skipping folder `{subfolder['title']}` @ {subfolder['alternateLink']}"
            )
            continue
        results.append(
            {
                "id": subfolder["id"],
                "title": subfolder["title"],
                "url": subfolder["alternateLink"],
                "person": subfolder["ownerNames"][0],
                "task_id": int(subfolder["title"].split(" @ ")[0]),
            }
        )
    return results


def init_drive(path_to_client_secrets_json: str):
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(path_to_client_secrets_json)
    gauth.LoadCredentialsFile("credentials.json")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("credentials.json")
    return GoogleDrive(gauth)


def _load_tasks_helper_parallel(args) -> Task:
    task, path_to_client_secrets_json = args
    # Load SOP + trace if demo folder exists in Gdrive
    if task.gdrive_folder is not None:
        drive = init_drive(path_to_client_secrets_json)
        task.get_sop(drive)
        task.get_trace(drive)
    return task


def load_tasks(
    path_to_drive_client_secrets_json: str,
    path_to_cache_dir: Optional[str] = None,
    path_to_tasks_xlsx: Optional[
        str
    ] = "./data/Process Mining Task Demonstrations.xlsx",
) -> List[Task]:
    """Load Tasks, merging from both Google Drive and .xlsx file.
    Requires one arg, `path_to_drive_client_secrets_json`, which is the path to the client secrets JSON file for Google Drive API.

    NOTE: This lazy loads tasks, so you need to explicitly call each `get()` method to fetch the data for SOP, trace, screenshots, etc.
    """
    # Load .xlsx file
    tasks: List[Task] = []
    df: pd.DataFrame = load_tasks_excel(path_to_tasks_xlsx)
    for row_idx, row in tqdm(df.iterrows(), "Loading .xslx..."):
        task = Task()
        task.task_id = row["Task ID"]
        task.person = row["Person"]
        task.gdrive_link = row["Gdrive link"]
        task.possible = row["Possible?"]
        task.difficulty = row["Difficulty"]
        task.notes = None if pd.isna(row["Notes"]) else row["Notes"]
        task.site = row["Site"]
        task.task = row["Task"]
        task.intent_template = row["Intent Template"]
        task.is_na = row["Is NA?"] == "TRUE"
        tasks.append(task)

    # Load tasks from Google Drive
    if path_to_cache_dir and os.path.exists(
        os.path.join(path_to_cache_dir, "tasks.pkl")
    ):
        print("Loading tasks from cache...")
        tasks: List[Task] = pickle.load(
            open(os.path.join(path_to_cache_dir, "tasks.pkl"), "rb")
        )
    else:
        n_procs: int = 10
        print("Downloading tasks from Google Drive...")
        demo_folders: List[Dict[str, str]] = load_demo_folders_from_drive(
            path_to_drive_client_secrets_json
        )

        # Match Gdrive folders to tasks
        for task in tqdm(tasks, desc="Matching tasks to Gdrive folders..."):
            for demo_folder in demo_folders:
                if (
                    demo_folder["task_id"] == task.task_id
                    and demo_folder["person"] == task.person
                ):
                    task.gdrive_folder = {
                        "id": demo_folder["id"],
                        "title": demo_folder["title"],
                        "url": demo_folder["url"],
                        "person": demo_folder["person"],
                        "task_id": demo_folder["task_id"],
                    }
                    break
        print(
            f"# of tasks matched with demo folders: {len([ x for x in tasks if x.gdrive_folder is not None ])} out of {len(tasks)} tasks"
        )

        # Load SOP.txt and trace.json in parallel
        os.makedirs(path_to_cache_dir, exist_ok=True)
        with Pool(n_procs) as pool:
            tasks = list(
                tqdm(
                    pool.imap(
                        _load_tasks_helper_parallel,
                        [(task, path_to_drive_client_secrets_json) for task in tasks],
                    ),
                    total=len(tasks),
                    desc="Downloading SOP and trace...",
                )
            )
        pickle.dump(tasks, open(os.path.join(path_to_cache_dir, "tasks.pkl"), "wb"))

    return tasks


def load_tasks_from_local(path_to_parent_dir: str) -> List[Task]:
    """Load Tasks from local folder"""
    # Load .xlsx file
    tasks: List[Task] = []
    metadata_json = json.load(
        open(os.path.join(path_to_parent_dir, "metadata.json"), "r")
    )["results"]
    folder_name_2_gdrive_link = {
        item["folder_name"]: item["url"] for item in metadata_json if item is not None
    }
    df_valid: pd.DataFrame = pd.read_csv(
        os.path.join(path_to_parent_dir, "df_valid.csv")
    )
    df_excel: pd.DataFrame = load_tasks_excel(
        os.path.join(path_to_parent_dir, "Process Mining Task Demonstrations.xlsx")
    )
    for row_idx, row in tqdm(df_valid.iterrows(), "Loading df_valid.csv..."):
        # Match row to df_excel
        df_excel_row = df_excel[
            (df_excel["Gdrive link"] == folder_name_2_gdrive_link[row["folder_name"]])
            & (df_excel["Task ID"] == row["task_id"])
        ]
        assert (
            len(df_excel_row) == 1
        ), f"The folder_name {row['folder_name']} matched with {len(df_excel_row)} rows in df_excel, but expected exactly 1. gdrive_link: {folder_name_2_gdrive_link[row['folder_name']]}"
        task = Task()
        task.task_id = row["task_id"]
        task.person = df_excel_row["Person"].values[0]
        task.gdrive_link = df_excel_row["Gdrive link"].values[0]
        task.possible = df_excel_row["Possible?"].values[0]
        task.difficulty = df_excel_row["Difficulty"].values[0]
        task.site = df_excel_row["Site"].values[0]
        task.task = df_excel_row["Task"].values[0]
        task.intent_template = df_excel_row["Intent Template"].values[0]
        task.is_na = df_excel_row["Is NA?"].values[0] == "TRUE"
        task.sop = open(get_path_to_sop_txt(row["path_to_demo_folder"]), "r").read()
        task.path_to_video = os.path.join(
            row["path_to_demo_folder"],
            find_files_by_prefix_suffix(row["path_to_demo_folder"], suffix=".mp4")[0],
        )
        task.trace = json.load(
            open(get_path_to_trace_json(row["path_to_demo_folder"]), "r")
        )
        tasks.append(task)

    return tasks


def add_standard_experiment_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "path_to_input_dir",
        help="Path to folder containing all task folders.",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task ID to address",
    )
    parser.add_argument(
        "--path_to_output_dir",
        default="./outputs/",
        type=str,
        required=False,
        help="Path to output directory",
    )
    return parser


def build_prompt_s_a_sequence(
    trace: Dict[str, Any], path_to_screenshots: str
) -> Tuple[Dict[str, Any], List[str]]:
    # Loop through trace, interleaving screenshots (states) and actions
    prompt_s_a_sequence: List[str] = []
    paths_to_screenshots: List[str] = []
    for item in trace:
        if item["type"] == "state":
            path_to_screenshot, encoded_image = load_screenshot_for_state(
                item, path_to_screenshots
            )
            paths_to_screenshots.append(path_to_screenshot)
            prompt_s_a_sequence.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        }
                    ],
                }
            )
        elif item["type"] == "action":
            action: str = convert_trace_action_to_dsl(item)["action"]
            prompt_s_a_sequence.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Action: {action}",
                        }
                    ],
                }
            )
        else:
            raise Exception(f"Unknown type for `item` in `s_a_sequence`: {type(item)}")
    return prompt_s_a_sequence, paths_to_screenshots


def add_standard_experiment_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "path_to_input_dir",
        help="Path to folder containing all task folders.",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="Task ID to address",
    )
    parser.add_argument(
        "--path_to_output_dir",
        default="./outputs/",
        type=str,
        required=False,
        help="Path to output directory",
    )
    return parser


def get_task_ids_in_folder(path_to_input_dir: str) -> List[int]:
    # Given `path_to_input_dir`, returns a list of unique task IDs across all demos
    task_ids: List[int] = list(
        set(
            [
                int(x.split(" @ ")[0])
                for x in os.listdir(path_to_input_dir)
                if " @ " in x
            ]
        )
    )
    return task_ids


def get_folders_for_task_id(path_to_input_dir: str, task_id: int) -> List[str]:
    # Identify folders corresponding to this `task_id` in `path_to_input_dir`
    path_to_task_folders: List[str] = []
    for folder in os.listdir(path_to_input_dir):
        if folder.startswith(f"{task_id} @ "):
            path_to_task_folders.append(os.path.join(path_to_input_dir, folder))
    assert (
        len(path_to_task_folders) > 0
    ), f"Could not find any folders corresponding to task_id {task_id} in {path_to_input_dir}"
    return path_to_task_folders


def get_best_k_folders_for_task_id(
    path_to_input_dir: str, task_id: int, k: int
) -> List[str]:
    """Given a task_id, return the k best folders from `path_to_input_dir` based on the ranks in the Gold SOPs sheet."""

    df: Dict[str, pd.DataFrame] = pd.read_excel(
        get_rel_path(__file__, "../data/Process Mining Task Demonstrations.xlsx"),
        sheet_name=None,
    )
    metadata_json = json.load(
        open(os.path.join(get_rel_path(__file__, "../data/metadata.json")), "r")
    )["results"]
    gdrive_link_2_folder_name = {
        item["url"]: item["folder_name"] for item in metadata_json if item is not None
    }
    gold_sops_df: pd.DataFrame = df["Gold SOPs"]
    task_row: pd.DataFrame = gold_sops_df[gold_sops_df["Task ID"] == task_id]

    assert (
        len(task_row) == 1
    ), f"Could not find a unique row for task_id {task_id} in the Gold SOPs sheet"

    # Get ranks for each demo from Gold SOPs sheet
    task_row: pd.Series = task_row.iloc[0]
    ranks: Dict[str, int] = dict()
    for i in range(1, 6):
        col_name: str = f"Sheet {i} Demo"
        if pd.isna(task_row[col_name]):
            continue
        ranks[f"Sheet {i}"] = int(task_row[col_name])

    if k is not None and k > len(ranks):
        raise ValueError(
            f"Cannot get {k} best demos for task_id {task_id} because only {len(ranks)} demos have ranks"
        )

    # Sort by rank
    best_demos: List[str] = [
        sheet_name for sheet_name, _ in sorted(ranks.items(), key=lambda item: item[1])
    ]
    if k is not None:
        best_demos = best_demos[:k]
    path_to_task_folders: List[str] = []

    # Get gdrive link for each best demo and convert to folder name
    for sheet_name in best_demos:
        sheet_task_row: pd.DataFrame = df[sheet_name][
            df[sheet_name]["Task ID"] == task_id
        ]

        assert (
            len(sheet_task_row) == 1
        ), f"Could not find a unique row for task_id {task_id} in {sheet_name}"

        gdrive_link: str = sheet_task_row.iloc[0]["Gdrive link"]
        path_to_task_folders.append(
            os.path.join(path_to_input_dir, gdrive_link_2_folder_name[gdrive_link])
        )

    return path_to_task_folders


def get_path_to_trace_json(
    path_to_task_folder: str, is_no_assert: bool = False
) -> Optional[str]:
    files: List[str] = [
        x
        for x in find_files_by_prefix_suffix(path_to_task_folder, "", ".json")
        if not (x.startswith("[gt]") or x.startswith("[raw]"))
    ]
    if not is_no_assert:
        assert (
            len(files) > 0
        ), f"Could not find any trace.json files in {path_to_task_folder}"
    return os.path.join(path_to_task_folder, files[0]) if len(files) > 0 else None


def get_path_to_gt_trace_json(
    path_to_task_folder: str, is_no_assert: bool = False
) -> Optional[str]:
    files: List[str] = [
        x
        for x in find_files_by_prefix_suffix(path_to_task_folder, "", ".json")
        if x.startswith("[gt]")
    ]
    if not is_no_assert:
        assert (
            len(files) > 0
        ), f"Could not find any [gt] trace.json files in {path_to_task_folder}"
    return os.path.join(path_to_task_folder, files[0]) if len(files) > 0 else None


def get_path_to_sop_txt(
    path_to_task_folder: str, is_no_assert: bool = False
) -> Optional[str]:
    files: List[str] = find_files_by_prefix_suffix(path_to_task_folder, "SOP", ".txt")
    if not is_no_assert:
        assert (
            len(files) > 0
        ), f"Could not find any SOP.txt files in {path_to_task_folder}"
    return os.path.join(path_to_task_folder, files[0]) if len(files) > 0 else None


def get_path_to_screenshots_dir(
    path_to_task_folder: str, is_no_assert: bool = False
) -> str:
    file: str = os.path.join(path_to_task_folder, "screenshots/")
    if not is_no_assert:
        assert os.path.isdir(
            file
        ), f"Could not find screenshots directory in {path_to_task_folder}"
    return file


def find_files_by_prefix_suffix(directory, prefix="", suffix=".txt") -> List[str]:
    """Returns file name, not path"""
    matching_files = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(suffix):
            matching_files.append(file)
    return matching_files
