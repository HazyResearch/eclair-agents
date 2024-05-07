#!/bin/bash

# Usage: ./run_experiments.sh <PATH_TO_EXPERIMENT_DATA_DIR>

BASE_DIR=$1

# Loop through all folders in BASE_DIR
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' folder; do
    # Check if folder is a directory
    if [ -d "$folder" ]; then
        # Run the sop_generation.py script on the folder
        python evaluate.py "$folder" --is_actuation
        python evaluate.py "$folder" --is_actuation --is_include_sop
        python evaluate.py "$folder" --is_action
        python evaluate.py "$folder" --is_action --is_include_sop
    fi
done
