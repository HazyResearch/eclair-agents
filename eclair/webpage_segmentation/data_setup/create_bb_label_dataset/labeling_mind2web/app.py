# from flask import Flask, jsonify, request
import pandas as pd
import os
import random
from flask import Flask, render_template, request, jsonify
import ast
from PIL import Image
import pickle

app = Flask(__name__)

ROBOFLOW = "ROBOFLOW"
MIND2WEB = "MIND2WEB"
LABEL_SOURCE = MIND2WEB #"ROBOFLOW" #"ALL"

webui_test_set_dir = "/media/nvme_data/ehr_workflows/hf_home/hub/webui-7k-yolo-inner-box/test"
images_dir = os.path.join(webui_test_set_dir,"images")
labels_dir = os.path.join(webui_test_set_dir,"labels")

image_filenames = os.listdir(os.path.join(webui_test_set_dir,"images"))
# dataset_path = '/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/bb_labels.csv'
dataset_path = '/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/bb_labels_mind2web.csv'
labels_path = '/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/bbs'
import random

global IMAGE_BB_DF

@app.route('/get_first_image_current_task', methods=['GET'])
def get_first_image_current_task():
    global current_image_index, IMAGE_BB_DF
    current_task_id = IMAGE_BB_DF.iloc[current_image_index]['task_id']
    new_image_index = current_image_index - 1
    while IMAGE_BB_DF.iloc[new_image_index]["task_id"] == current_task_id:
        new_image_index -= 1
    current_image_index = new_image_index + 1
    return get_image_data_by_index(current_image_index) 

@app.route('/get_first_image_next_task', methods=['GET'])
def get_first_image_next_task():
    global current_image_index, IMAGE_BB_DF
    current_task_id = IMAGE_BB_DF.iloc[current_image_index]['task_id']
    new_image_index = current_image_index + 1
    while IMAGE_BB_DF.iloc[new_image_index]["task_id"] == current_task_id:
        new_image_index += 1
    current_image_index = new_image_index
    return get_image_data_by_index(current_image_index)


@app.route('/get_previous_image', methods=['GET'])
def get_previous_image():
    global current_image_index
    current_image_index = max(0, current_image_index - 1)  # Prevent negative index
    return get_image_data_by_index(current_image_index)

@app.route('/get_image', methods=['GET'])
def get_image():
    global current_image_index
    current_image_index += 1  # Move to the next image
    return get_image_data_by_index(current_image_index)

def get_image_data_by_index(index):
    global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX

    # Ensure the index is within the bounds of available images
    if index < 0 or index >= len(image_filenames):
        return jsonify({"error": "Image index out of bounds"}), 404

    row = IMAGE_BB_DF.iloc[current_image_index]


    # Get the image path 
    image_path = "/media/nvme_data/eclair_agents/Mind2Web/" + row["path_to_screenshot_image"]

    # Get any existing image label from our df 
    df_row = df[df['image_path'] == image_path]
    label = df_row['description_label'].iloc[0] if not df_row.empty else ''

    # Extract the bounding box data
    unscaled_bb = [0.0]
    unscaled_bb.extend(ast.literal_eval(row["element_bounding_box_yolo_format"]))
    unscaled_bb = [float(bb_i) for bb_i in unscaled_bb]

    normalized_bb = normalize_bounding_box(unscaled_bb, image_path)

    # Construct the web-accessible path to the image
    web_accessible_path = f'/static/mind2web_images/screenshots/{os.path.basename(image_path)}'

    # Save these variables for future global access
    BOUNDING_BOX = normalized_bb
    IMAGE_PATH = image_path

    # Pickle the contents of the label path, which just contains labels 
    all_labels = row["path_to_all_groundtruth_bounding_boxes_of_page"]
    all_labels_path = f'{labels_path}/{os.path.basename(row["path_to_screenshot_image"]).replace(".jpeg",".pickle")}'

    os.makedirs(labels_path, exist_ok=True)

    # TODO: pickle all_labels
    with open(all_labels_path, 'wb') as file:
        pickle.dump(all_labels, file)

    LABEL_PATH = all_labels_path

    print(f"Existing label: {label}")

    return jsonify({'imagePath': web_accessible_path, 'boundingBox': normalized_bb, 'label': label})

    # return jsonify({'imagePath': web_accessible_path, 'boundingBox': normalized_bb})


def normalize_bounding_box(bb, image_path):
    # Load the image to get its dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Convert bounding box values to float and extract class_id
    class_id, x_left, y_top, width_bb, height_bb = map(float, bb)

    # Normalize the values
    width_bb_normalized = width_bb / img_width
    height_bb_normalized = height_bb / img_height
    x_center_normalized = x_left / img_width + width_bb_normalized / 2
    y_center_normalized = y_top / img_height + height_bb_normalized / 2

    return [class_id, x_center_normalized, y_center_normalized, width_bb_normalized, height_bb_normalized]

def select_new_image(seen_images, all_images):
    suggested_image = ""
    while (suggested_image == "") or (suggested_image in seen_images):
        print(f"len(all_images) {len(all_images)}")
        print(f"len(all_images) {random.randint(0, len(all_images))}")
        suggested_image = all_images[random.randint(0, len(all_images))]
    
    return suggested_image

def read_yolo_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Remove any whitespace or newline characters
        bounding_boxes = [line.strip() for line in lines if line.strip()]
        return bounding_boxes

def choose_random_bounding_box(file_path):
    bounding_boxes = read_yolo_bounding_boxes(file_path)
    if bounding_boxes:
        return random.choice(bounding_boxes)
    else:
        return None

@app.route('/submit_label', methods=['POST'])
def submit_label():
    global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX, df
    # global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX
    label_data = request.json
    # image_path = label_data['imagePath']

    print("Received label:", label_data['label'])
    print(f"bounding_box {BOUNDING_BOX}")

    image_basename = os.path.basename(IMAGE_PATH)
    df_row_index = df[df['image_path'].apply(lambda x: os.path.basename(x)) == image_basename].index.tolist()

    # If row already exists, just overwrite it
    if df_row_index:
        df.at[df_row_index[0], 'description_label'] = label_data['label']
        df.at[df_row_index[0], 'element_type'] = label_data['elementType']
    # Otherwise, write a new row
    else:
        new_row = {
            'description_label': label_data['label'],
            'element_type': label_data['elementType'],
            'image_path': IMAGE_PATH,
            'bb': BOUNDING_BOX,
            'bb_x': BOUNDING_BOX[1],
            'bb_y': BOUNDING_BOX[2],
            'bb_width': BOUNDING_BOX[3],
            'bb_height': BOUNDING_BOX[4],
            'label_path': LABEL_PATH,
        }
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_csv(dataset_path, index=False)

    # Remember to save the DataFrame to CSV if needed
    return jsonify({"status": "success"})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX, df, current_image_index
    if not os.path.exists(dataset_path):
        columns = ['image_path', 'label_path', 'description_label', 'element_type', 'bb', 'bb_x', 'bb_y', 'bb_width', 'bb_height']
        df = pd.DataFrame(columns=columns)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)
    
    # Load IMAGE_BB_DF
    if LABEL_SOURCE == MIND2WEB:
        IMAGE_BB_DF = pd.read_csv("/media/nvme_data/eclair_agents/Mind2Web/new_dataset/annotations.csv")
        IMAGE_BB_DF["task_id"] = IMAGE_BB_DF['path_to_screenshot_image'].str.split('-').str[0].str.split('/').str[-1]

    else:
        IMAGE_BB_DF = None  # Or handle the ROBOFLOW case

    # Initialize current_image_index based on the latest image_path in df
    if not df.empty:
        last_image_path = df.iloc[-1]['image_path']
        image_base_name = os.path.basename(last_image_path).replace(".jpeg","")
        if image_base_name in IMAGE_BB_DF['row_name'].values:
            current_image_index = IMAGE_BB_DF[IMAGE_BB_DF['row_name'] == image_base_name].index[0] + 1
        else:
            current_image_index = 0
    else:
        current_image_index = 0

    IMAGE_PATH = ""
    LABEL_PATH = ""
    BOUNDING_BOX = []
    app.run(debug=True)
