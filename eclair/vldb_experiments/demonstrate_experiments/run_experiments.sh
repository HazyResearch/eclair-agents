#!/bin/bash

# Usage: ./run_experiments.sh ../data

BASE_DIR=$1

# Loop through all folders in BASE_DIR
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' folder; do
    # Check if folder is a directory
    if [ -d "$folder" ]; then
        # Run the evaluate.py script on the folder
        echo $folder
        python evaluate.py "$folder" --is_td
        python evaluate.py "$folder" --is_td_kf
        python evaluate.py "$folder" --is_td_kf_act
    fi
done