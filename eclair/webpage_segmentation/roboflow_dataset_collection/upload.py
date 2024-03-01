'''Uploads a folder of images/annotations to RoboFlow'''
from roboflow import Roboflow
import argparse
from pathlib import Path

ROBOFLOW_API_KEY = "82ZUdIOMDqaJ6cW8VLBG"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload images to Roboflow")
    parser.add_argument("--path_to_folder", type=str, required=True, help="Path to folder containing images/annotations")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize Roboflow client
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    rf_project = rf.workspace("workflowaugmentation").project("workflowaugmentation/webpages-abgy4")
    
    # For each .png image...
    for file_path in Path(args.path_to_folder).glob('*.png'):
        if file_path.is_file():
            path_to_img: str = file_path
            path_to_annotation: str = file_path.with_suffix('.xml')

            # Convert XML's encoding from latin-1 to utf-8 for Roboflow
            with open(path_to_annotation, 'r', encoding='latin-1') as file:
                content = file.read()
            with open(path_to_annotation, 'w', encoding='utf-8') as file:
                file.write(content)

            try:
                rf_project.upload(str(path_to_img), str(path_to_annotation), split='train')
            except Exception as e:
                print(f"Error uploading {path_to_img}: {e}")
                continue
