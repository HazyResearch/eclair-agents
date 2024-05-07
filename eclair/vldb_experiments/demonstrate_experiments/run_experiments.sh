#!/bin/bash

# Construct relative path to dataset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath "${SCRIPT_DIR}/../../../data/vldb_experiments/demos")"

# Loop through all folders in BASE_DIR
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' folder; do
    # Check if folder is a directory
    if [ -d "$folder" ]; then
        # Run the evaluate.py script on the folder
        echo $folder
        python "${SCRIPT_DIR}/evaluate.py" "$folder" --is_td
        python "${SCRIPT_DIR}/evaluate.py" "$folder" --is_td_kf
        python "${SCRIPT_DIR}/evaluate.py" "$folder" --is_td_kf_act
    fi
done