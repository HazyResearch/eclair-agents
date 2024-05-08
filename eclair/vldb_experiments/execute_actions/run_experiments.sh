#!/bin/bash

# Construct relative path to dataset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(realpath "${SCRIPT_DIR}/../../../data/vldb_experiments/demos")"

# Loop through all folders in BASE_DIR
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' folder; do
    # Check if folder is a directory
    if [ -d "$folder" ]; then
        # Run the sop_generation.py script on the folder
        echo $folder
        python "${SCRIPT_DIR}/evaluate.py" "$folder" --is_actuation
        python "${SCRIPT_DIR}/evaluate.py" "$folder" --is_actuation --is_include_sop
        python "${SCRIPT_DIR}/evaluate.py" "$folder" --is_action
        python "${SCRIPT_DIR}/evaluate.py" "$folder" --is_action --is_include_sop
    fi
done
