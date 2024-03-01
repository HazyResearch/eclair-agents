# from flask import Flask, jsonify, request
import pandas as pd
import os
import random
from flask import Flask, render_template, request, jsonify
import ast
from PIL import Image
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

import random

global IMAGE_BB_DF

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

def get_random_image():
    global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX, df
    # global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX
    # Sample code to select a random image and its bounding box
    if LABEL_SOURCE == MIND2WEB:
        # image_bb_df = pd.read_csv("/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/annotations.csv")
        for index, row in IMAGE_BB_DF.iterrows():
            print(f"i {index}")
            original_img_path = "/media/nvme_data/eclair_agents/Mind2Web/" + row["path_to_screenshot_image"]
            web_accessible_path = f'/static/mind2web_images/screenshots/{os.path.basename(original_img_path)}'
            bb = [0.0]
            bb.extend(ast.literal_eval(row["element_bounding_box_yolo_format"]))

            IMAGE_PATH = original_img_path
            LABEL_PATH = row["path_to_all_groundtruth_bounding_boxes_of_page"] #"TODO"  # Replace with actual label path if necessary


            # Check if the image is not already in df
            if IMAGE_PATH not in df['image_path'].values:
                BOUNDING_BOX = bb
                print(f"Selected web_accessible_path: {web_accessible_path}")
                print(f"Selected image path: {IMAGE_PATH}")
                print(f"Selected image with bounding box: {bb}")

                bb = [float(bb_i) for bb_i in bb]

                BOUNDING_BOX = normalize_bounding_box(bb, original_img_path)
                print(f"bounding_box {BOUNDING_BOX}")


                # print(f"image_path {web_accessible_path}")
                # print(f"boundingBox {bb}")
                # print(f"boundingBox {type(bb)}")
                # print(f"boundingBox {bb[0]}")
                # print(f"boundingBox {type(bb[0])}")
                # print(f"returning.")
                return {'imagePath': web_accessible_path, 'boundingBox': BOUNDING_BOX}
                # break
                # return  # Exit the function once the next image is found

        print("No more new images available in the dataset.")
        # while True:
        #     # Select a random index from the DataFrame
        #     random_index = random.choice(image_bb_df.index)
        #     row = image_bb_df.iloc[random_index]

        #     original_img_path = "eclair-agents/eval/Mind2Web/" + row["path_to_screenshot_image"]
        #     # breakpoint()

        #     # id_folder_name = original_img_path.split(os.sep)[-2]  # Get the ID folder name
        #     #TODO: web path
        #     web_accessible_path = f'/static/mind2web_images/screenshots/{os.path.basename(original_img_path)}'
        #     bb = [0.0]
        #     bb.extend(ast.literal_eval(row["element_bounding_box_yolo_format"]))

        #     IMAGE_PATH = original_img_path #f"/media/nvme_data/ehr_workflows/hf_home/hub/webui-7k-yolo-inner-box/test/images/{id_folder_name}_{os.path.basename(original_img_path).replace('-screenshot-full.webp', '.jpg')}"
        #     LABEL_PATH = "TODO"#IMAGE_PATH.replace("images","labels").replace(".jpg",".txt")
        #     print(f"bb {bb}")

        #     # Check if the image is not already in df
        #     if IMAGE_PATH not in df['image_path'].values:
        #         break  # If it's not in df, break the loop and use this image

    elif LABEL_SOURCE == ROBOFLOW:
        image_bb_df = pd.read_csv("/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/data_setup/create_bb_label_dataset/focusable_elements_data.csv")
        # for i,row in image_bb_df.iterrows():
        #     IMAGE_PATH = row["Image File"]
        #     id_folder_name = IMAGE_PATH.split(os.sep)[-2]  # -1 is the file, -2 is the parent folder, -3 is the 'id' folder
        #     web_accessible_path = f'/static/images/{id_folder_name}_{os.path.basename(IMAGE_PATH).replace("-screenshot-full.webp",".jpg")}'
        #     bb = row["YOLO BB"]
            
        #     # We don't want to use an image that's already been saved
        #     if IMAGE_PATH in df['image_path'].values:
        #         continue
        while True:
            # Select a random index from the DataFrame
            random_index = random.choice(image_bb_df.index)
            row = image_bb_df.iloc[random_index]

            original_img_path = row["Image File"]

            id_folder_name = original_img_path.split(os.sep)[-2]  # Get the ID folder name
            web_accessible_path = f'/static/images/{id_folder_name}_{os.path.basename(original_img_path).replace("-screenshot-full.webp", ".jpg")}'
            bb = [0.0]
            bb.extend(ast.literal_eval(row["YOLO BB"]))

            IMAGE_PATH = f"/media/nvme_data/ehr_workflows/hf_home/hub/webui-7k-yolo-inner-box/test/images/{id_folder_name}_{os.path.basename(original_img_path).replace('-screenshot-full.webp', '.jpg')}"
            LABEL_PATH = IMAGE_PATH.replace("images","labels").replace(".jpg",".txt")
            print(f"bb {bb}")

            # Check if the image is not already in df
            if IMAGE_PATH not in df['image_path'].values:
                break  # If it's not in df, break the loop and use this image
    else:
        image_not_set = True

        while image_not_set:
            try:
                seen_images = []
                image_filename = select_new_image(seen_images, image_filenames)
                image_path = os.path.join(webui_test_set_dir,"images",image_filename)
                IMAGE_PATH = image_path
                # image_filename = "1655885042728_default_1280-720.jpg"
                web_accessible_path = f'/static/images/{image_filename}'
                # IMAGE_PATH = "/media/nvme_data/ehr_workflows/hf_home/hub/webui-7k-yolo-inner-box/test/images/1655885042728_default_1280-720.jpg"
                # image_path = "/media/nvme_data/ehr_workflows/hf_home/hub/webui-7k-yolo-inner-box/test/images/1655885042728_default_1280-720.jpg"
                # get the associated bounding box 
                label_filename = image_filename.replace(".jpg",".txt")
                label_path = os.path.join(webui_test_set_dir,"labels",label_filename)
                LABEL_PATH = label_path
                bb = choose_random_bounding_box(label_path).split()
                image_not_set = False
            except Exception as e:
                print(f"e {e}")

    bb = [float(bb_i) for bb_i in bb]

    BOUNDING_BOX = bb
    print(f"bounding_box {BOUNDING_BOX}")


    print(f"image_path {web_accessible_path}")
    print(f"boundingBox {bb}")

    print(f"boundingBox {type(bb)}")
    print(f"boundingBox {bb[0]}")
    print(f"boundingBox {type(bb[0])}")
    # breakpoint()

    return {'imagePath': web_accessible_path, 'boundingBox': bb}

    # image = random.choice(df['image_path'].tolist())
    # label_path = df[df['image_path'] == image]['label_path'].iloc[0]
    # with open(label_path, 'r') as file:
    #     lines = file.readlines()
    #     if lines:
    #         bounding_box = lines[0].strip().split()
    # return {'imagePath': image, 'boundingBox': bounding_box}

@app.route('/')
def index():
    return render_template('index.html')
    # return app.send_static_file('index.html')

@app.route('/get_image', methods=['GET'])
def get_image():
    image_data = get_random_image()
    return jsonify(image_data)

@app.route('/submit_label', methods=['POST'])
def submit_label():
    global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX, df
    # global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX
    label_data = request.json
    print("Received label:", label_data['label'])
    print(f"bounding_box {BOUNDING_BOX}")
    # Update the DataFrame here
    # new_row = {'image_path': IMAGE_PATH, 'label_path': LABEL_PATH, 'natural_language_descrip': label_data['label'], 'bb': BOUNDING_BOX, 'bb_x': BOUNDING_BOX[1],'bb_y': BOUNDING_BOX[2],'bb_width': BOUNDING_BOX[3],'bb_height': BOUNDING_BOX[4]}

    # Append the new row
    # df = df.append(new_row, ignore_index=True)

    df.to_csv(dataset_path)

    new_row = {
        'image_path': IMAGE_PATH,
        'label_path': LABEL_PATH,
        'description_label': label_data['label'],
        'element_type': label_data['elementType'],
        'bb': BOUNDING_BOX,
        'bb_x': BOUNDING_BOX[1],
        'bb_y': BOUNDING_BOX[2],
        'bb_width': BOUNDING_BOX[3],
        'bb_height': BOUNDING_BOX[4]
    }

    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_csv(dataset_path, index=False)

    # Remember to save the DataFrame to CSV if needed
    return jsonify({"status": "success"})

if __name__ == '__main__':
    global IMAGE_PATH, LABEL_PATH, BOUNDING_BOX, df
    if not os.path.exists(dataset_path):
        columns = ['image_path', 'label_path', 'description_label', 'element_type', 'bb', 'bb_x', 'bb_y', 'bb_width', 'bb_height']
        df = pd.DataFrame(columns=columns)
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)

    # Load image_bb_df here
    if LABEL_SOURCE == MIND2WEB:
        IMAGE_BB_DF = pd.read_csv("/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/Mind2Web/annotations.csv")
    else:
        IMAGE_BB_DF = None  # Or handle ROBOFLOW case

    IMAGE_PATH = ""
    LABEL_PATH = ""
    BOUNDING_BOX = []
    app.run(debug=True)
