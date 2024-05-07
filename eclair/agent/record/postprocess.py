"""
Usage:

python postprocess.py './data/Nursing/sitter @ 2024-03-21-10-08-24' --task_descrip 'Sitter order'
"""
import datetime
import shutil
import os
from eclair.utils.helpers import (
    convert_mousedown_mouseup_to_click,
    extract_screenshots_for_demo,
    get_path_to_screen_recording,
    get_path_to_screenshots_dir,
    get_path_to_trace_json,
    merge_consecutive_states,
)
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from typing import Dict, List, Optional, Any
from moviepy.editor import VideoFileClip
import json
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_demo_folder", type=str)
    parser.add_argument("--task_descrip", type=str, default='', help="Task description")
    parser.add_argument("--buffer_seconds", type=float, default=0.05, help="Buffer seconds to add to each state (adjust to align video with trace.json)")
    parser.add_argument("--valid_application_name", type=str, default='Citrix Viewer', help="If specified, filter out all states/actions that are not for this application")
    parser.add_argument("--is_verbose", action="store_true", help="If TRUE, then print out stuff")
    return parser.parse_args()

def clean_demo(path_to_demo_folder: str, valid_application_name: str = 'Citrix Viewer', buffer_seconds: float = 0.05):
    """Clean demo to align trace.json and screenshots with video recording."""
    # Get .json trace
    path_to_trace: str = get_path_to_trace_json(path_to_demo_folder)
    full_trace: Dict[str, str] = json.loads(open(path_to_trace, "r").read())
    trace_json: List[Dict[str, str]] = full_trace['trace']
    
    # Check if we've already cleaned this trace.json
    is_already_cleaned: bool = full_trace.get('is_cleaned', False)

    # Get screen reecording
    path_to_screen_recording: str = get_path_to_screen_recording(path_to_demo_folder)

    # Get screenshots
    path_to_screenshots_dir: str = get_path_to_screenshots_dir(path_to_demo_folder)
    screenshots: List[str] = os.listdir(path_to_screenshots_dir)
    screenshots = sorted(screenshots) # sort by filename, from 0 -> N
    
    
    ##############################
    ##############################
    # Group actions => semantic grouping (e.g. mousedown+mouseup => click)
    ##############################
    ##############################
    trace_json = convert_mousedown_mouseup_to_click(trace_json)
    trace_json = merge_consecutive_states(trace_json)
    
    ##############################
    ##############################
    # Truncate video
    ##############################
    ##############################
    
    # Align video end with expected video length
    expected_video_secs_length: float = trace_json[-1]['data']['secs_from_start']
    actual_video_secs_length: float = VideoFileClip(path_to_screen_recording).duration
    diff_secs: float = expected_video_secs_length - actual_video_secs_length
    if diff_secs > 0:
        diff_secs -= buffer_seconds # add 0.05s buffer
    # Ignore if already cleaned
    if is_already_cleaned:
        diff_secs = 0

    # Adjust all timestamps earlier by diff_secs so that last state's timestamp aligns with last video frame
    for event in trace_json:
        event['data']['secs_from_start'] -= diff_secs
        event['data']['timestamp'] = (datetime.datetime.fromisoformat(event['data']['timestamp']) - datetime.timedelta(seconds=diff_secs)).isoformat()
        if 'start_timestamp' in event['data']:
            event['data']['start_timestamp'] = (datetime.datetime.fromisoformat(event['data']['start_timestamp']) - datetime.timedelta(seconds=diff_secs)).isoformat()
        if 'end_timestamp' in event['data']:
            event['data']['end_timestamp'] = (datetime.datetime.fromisoformat(event['data']['end_timestamp']) - datetime.timedelta(seconds=diff_secs)).isoformat()

    # Drop any states/actions that occurred before video started recording (i.e. timestamp < 0)
    trace_json = [event for event in trace_json if event['data']['secs_from_start'] >= 0]

    # Truncate from start
    valid_start_idx: int = 0
    for event in trace_json:
        if event['type'] == 'state':
            active_application_name: str = event['data']['active_application_name']
            if active_application_name != valid_application_name:
                valid_start_idx += 2 # skip this state + action
            else:
                break
    # Truncate from end
    valid_end_idx: int = len(trace_json)
    for event in trace_json[::-1]:
        if event['type'] == 'state':
            active_application_name: str = event['data']['active_application_name']
            if active_application_name != valid_application_name:
                valid_end_idx -= 2 # skip this state + action
            else:
                break
    
    # Determine new video START time
    valid_start_secs_from_start: float = 0
    if valid_start_idx > 0:
        valid_start_secs_from_start = trace_json[valid_start_idx]['data']['secs_from_start']

    # Determine new video END time
    valid_end_secs_from_start: Optional[float] = None
    if valid_end_idx < len(trace_json):
        valid_end_secs_from_start = trace_json[valid_end_idx - 1]['data']['secs_from_start']
    
    # Ignore if already cleaned
    if is_already_cleaned:
        valid_start_secs_from_start = 0
        valid_end_secs_from_start = None

    # Adjust all timestamps to be relative to new video START time
    for event in trace_json:
        event['data']['secs_from_start'] -= valid_start_secs_from_start
        event['data']['timestamp'] = (datetime.datetime.fromisoformat(event['data']['timestamp']) - datetime.timedelta(seconds=valid_start_secs_from_start)).isoformat()
        if 'start_timestamp' in event['data']:
            event['data']['start_timestamp'] = (datetime.datetime.fromisoformat(event['data']['start_timestamp']) - datetime.timedelta(seconds=valid_start_secs_from_start)).isoformat()
        if 'end_timestamp' in event['data']:
            event['data']['end_timestamp'] = (datetime.datetime.fromisoformat(event['data']['end_timestamp']) - datetime.timedelta(seconds=valid_start_secs_from_start)).isoformat()

    # Truncate trace.json
    trace_json = trace_json[valid_start_idx:valid_end_idx]
    full_trace['trace'] = trace_json
    full_trace['task'] = {
        'description': task_descrip if task_descrip else full_trace.get('task', {}).get('description', None)
    }
    json.dump(full_trace, open(path_to_trace, "w"), indent=2)

    # Truncate video
    path_to_raw_file: str = os.path.join(os.path.dirname(path_to_screen_recording), "[raw] " + os.path.basename(path_to_screen_recording))
    if not os.path.exists(path_to_raw_file):
        shutil.copy(path_to_screen_recording, path_to_raw_file)
    if valid_start_secs_from_start > 0 or valid_end_secs_from_start is not None:
        path_to_tmp_file: str = os.path.join(os.path.dirname(path_to_screen_recording), "[new] " + os.path.basename(path_to_screen_recording))
        ffmpeg_extract_subclip(path_to_screen_recording, 
                               valid_start_secs_from_start, 
                               valid_end_secs_from_start if valid_end_secs_from_start else 99999999, 
                               targetname=path_to_tmp_file)
        os.remove(path_to_screen_recording)
        os.rename(path_to_tmp_file, path_to_screen_recording)

    ##############################
    ##############################
    # Resample screenshots
    ##############################
    ##############################
    
    extract_screenshots_for_demo(path_to_demo_folder)
    full_trace: Dict[str, str] = json.loads(open(path_to_trace, "r").read()) # need to reload trace.json to get updated screenshot names

    # Mark that we've successfully cleaned this trace.json
    full_trace['is_cleaned'] = True
    json.dump(full_trace, open(path_to_trace, "w"), indent=2)

if __name__ == "__main__":
    args = parse_args()
    path_to_demo_folder: str = args.path_to_demo_folder
    task_descrip: str = args.task_descrip
    valid_application_name: str = args.valid_application_name
    buffer_seconds: float = args.buffer_seconds
    is_verbose: bool = args.is_verbose
    
    clean_demo(path_to_demo_folder, valid_application_name=valid_application_name, buffer_seconds=buffer_seconds)