import os
import json
from tqdm import tqdm
import argparse
import yaml
import shutil

def remove_outer_boxes(input_dir, output_dir):

    def clean_and_filter_boxes(bounding_boxes):
        # Remove None entries and boxes with zero width or height
        return {k: v for k, v in bounding_boxes.items() if v is not None and v['width'] > 0 and v['height'] > 0}

    def process_and_save_file(file_path, output_dir):
        with open(file_path, 'r') as file:
            # print(f"Opening {file_path}.")
            bb_data = file.readlines()

        json_data = {}
        for idx, line in enumerate(bb_data):
            _, x, y, width, height = line.strip().split()
            bbox = {"x": float(x), "y": float(y), "width": float(width), "height": float(height)}
            json_data[int(idx)] = bbox

        boxes_to_remove = []

        # Assuming bounding_boxes is a list of dictionaries
        for box_a_id, box_a in json_data.items():
            for box_b_id, box_b in json_data.items():
                if box_a_id != box_b_id and box_a_id not in boxes_to_remove and box_b_id not in boxes_to_remove:
                    box_a_x_start = round(box_a['x'] - box_a['width'] / 2, 4)
                    box_a_x_end = round(box_a['x'] + box_a['width'] / 2, 4)
                    box_a_y_start = round(box_a['y'] - box_a['height'] / 2, 4)
                    box_a_y_end = round(box_a['y'] + box_a['height'] / 2, 4)

                    box_b_x_start = round(box_b['x'] - box_b['width'] / 2, 4)
                    box_b_x_end = round(box_b['x'] + box_b['width'] / 2, 4)
                    box_b_y_start = round(box_b['y'] - box_b['height'] / 2, 4)
                    box_b_y_end = round(box_b['y'] + box_b['height'] / 2, 4)

                    # If box B is within box A, remove box A
                    if box_a_x_start <= box_b_x_start and box_b_x_end <= box_a_x_end and box_a_y_start <= box_b_y_start and box_b_y_end <= box_a_y_end:
                        boxes_to_remove.append(box_a_id)

        boxes_to_remove = set(boxes_to_remove)
        for box_id in boxes_to_remove:
            json_data.pop(box_id, None)

        cleaned_data = clean_and_filter_boxes(json_data)

        # Write back to YOLO format
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_file_path, 'w') as output_file:
            for box in cleaned_data.values():
                line = f"0 {box['x']} {box['y']} {box['width']} {box['height']}\n"
                output_file.write(line)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # print(f"input_dir {input_dir}")
    for split in ["test","train","valid"]:
        split_output_dir_labels = os.path.join(output_dir,split,"labels")
        split_output_dir_images = os.path.join(output_dir,split,"images")
        os.makedirs(split_output_dir_labels, exist_ok=True)
        # os.makedirs(split_output_dir_images, exist_ok=True) 
    
        # First, copy over the image files
        print(f"Copying over images.")
        shutil.copytree(f"{input_dir}/{split}/images", split_output_dir_images)

        # Then, copy over the label files without bounding boxes
        split_input_dir_labels = f"{input_dir}/{split}/labels"
        file_paths = [os.path.join(split_input_dir_labels, f) for f in os.listdir(split_input_dir_labels) if f.endswith(".txt")]
        # print(f"file_paths {file_paths}")
        for file_path in tqdm(file_paths, desc=f"Removing outer bounding boxes from label files for {split}."):
            process_and_save_file(file_path, split_output_dir_labels)
    
    # Define the YAML content as a dictionary
    yaml_data = {
        'names': ['webelement'],
        'nc': 1,
        'test': '../test/images',
        'train': '../train/images',
        'val': '../valid/images'
    }

    # Specify the directory where you want to save the file
    file_path = os.path.join(output_dir, 'data.yaml')

    # Write the YAML data to the file
    with open(file_path, 'w') as file:
        yaml.dump(yaml_data, file, sort_keys=False)

    print(f"Dataset written to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Format WebUI data into YOLO format.')
    parser.add_argument('--version', type=str, help='Directory of yolo formatted webui data')
    # parser.add_argument('--output_dir', type=str, help='Directory of new data with outer boxes removed.')

    args = parser.parse_args()

    input_dir = os.path.join(os.getenv("HF_HOME"), f"hub/webui-{args.version}-yolo")
    output_dir = os.path.join(os.getenv("HF_HOME"), f"hub/webui-{args.version}-yolo-inner-box-10")

    # Directory paths
    remove_outer_boxes(input_dir,output_dir)
    
if __name__ == "__main__":
    main()