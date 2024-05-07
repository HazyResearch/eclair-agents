#!/bin/bash

# Cleans up traces generated from WebArena demos
# NOTE: You don't need to run this yourself -- included just for reference
python3 trace_truncate_non_chrome_apps.py
python3 trace_merge_scrolls.py
python3 trace_fix_coords.py
python3 screenshots_resample_keystrokes.py
python3 trace_add_webarena_task_json.py
python3 trace_add_gt_labels.py
python3 trace_force_add_gt_element.py