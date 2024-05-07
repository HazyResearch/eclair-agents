#!/bin/bash

# Usage: bash run_experiments.sh <PATH_TO_EXPERIMENT_DATA_DIR>

BASE_DIR=$1

# Loop through all folders in BASE_DIR
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' folder; do
    # Check if folder is a directory
    if [ -d "$folder" ]; then
        # Run the sop_generation.py script on the folder
        echo $folder
        python evaluate.py "$folder" --is_actuation
        python evaluate.py "$folder" --is_precondition
        python evaluate.py "$folder" --is_task_completion
        python evaluate.py "$folder" --is_task_trajectory
    fi
done