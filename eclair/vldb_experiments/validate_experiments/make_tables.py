"""
Usage:

python make_tables.py "/Users/mwornow/Dropbox/Stanford/Re Lab/Workflows/eclair-agents/eclair/vldb_experiments/validate_experiments/outputs"
"""
import collections
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_input_dir", type=str, help="Path to folder containing all demo subdirectories")
    parser.add_argument("--path_to_output_dir", type=str, default="./outputs", help="Path to output folder")
    return parser.parse_known_args()

def main(args):
    path_to_input_dir: str = args.path_to_input_dir
    path_to_output_dir: str = args.path_to_output_dir
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load all df's
    eval_2_dfs: Dict[List[pd.DataFrame]] = collections.defaultdict(list)
    for demo_folder in tqdm(os.listdir(path_to_input_dir), desc="Loading .csv's"):
        if ' @ ' not in demo_folder:
            continue
        path_to_demo_folder: str = os.path.join(path_to_input_dir, demo_folder)
        for file in os.listdir(path_to_demo_folder):
            if not file.endswith(".csv"):
                continue
            path_to_csv: str = os.path.join(path_to_demo_folder, file)
            eval_type: str = file.replace('.csv', '')
            df: pd.DataFrame = pd.read_csv(path_to_csv)
            df['demo'] = demo_folder
            eval_2_dfs[eval_type].append(df)

    # Merge all df's of same eval type
    eval_2_df: Dict[str, pd.DataFrame] = { eval_type: pd.concat(dfs) for eval_type, dfs in eval_2_dfs.items() }
    for eval_type, df in eval_2_df.items():
        # Save merged .csv
        df.to_csv(os.path.join(path_to_output_dir, f"validate_{eval_type}_merged.csv"), index=False)
        # Calc metrics
        tp = (df['is_correct'] & df['gt_is_met']).sum()
        tn = (df['is_correct'] & ~df['gt_is_met']).sum()
        fp = (~df['is_correct'] & df['gt_is_met']).sum()
        fn = (~df['is_correct'] & ~df['gt_is_met']).sum()
        # Print metrics
        print(f"==== {eval_type} ====")
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn)}")
        print(f"Sensitivity: {tp / (tp + fn)}") # P(y_hat=1|y=1) -- Given gt is true, how often is the model correct?
        print(f"Specificity: {tn / (tn + fp)}") # P(y_hat=0|y=0) -- Given gt is false, how often is the model correct?
        print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
        print(f"Total: {tp + tn + fp + fn} | GT=True: {tp + fn} | GT=False: {tn + fp}")
        assert tp + tn + fp + fn == df.shape[0]

if __name__ == "__main__":
    args, __ = parse_args()
    main(args)