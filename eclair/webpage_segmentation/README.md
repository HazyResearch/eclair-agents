# Segmentation / Bounding Box Model Training
**Goal:** Train a model to segment relevant UI elements of a digital interface (web, desktop, mobile, etc.)

## Folder Structure

TODO

* `models/`
* `datasets/`
* `roboflow/`

## Data

To download our custom-generated dataset from Roboflow and format it, run:

```bash
cd data_setup
python3 download_roboflow.py
```

To download WebUI, run:

```bash
cd data_setup
python3 download_webui_hf.py --version 7k
```

This will create a folder with WebUI data at `<HF_HOME>/hub/webui-7k`.

If you want to download a larger version of the dataset, use `--version 70k` or `--version 350k`.

To convert the WebUI datasest for use with any of the YOLO models, run:

```bash
cd data_setup
python3 convert_webui_to_yolo.py --version <version>
```

This will create a new folder with the formatted data at `<HF_HOME>/hub/webui-7k-yolo`.

To remove all outer bounding boxes from this yolo formatted WebUI dataset, run:

```bash
cd data_setup
python3 format_inner_boxes.py --version <version>
```

#### WebUI

Folder structure:

**default_1920-1080-box.json** is a 1920x1080 resolution screenshot with box models for each UI element. A box model includes information on the margin, padding, and border of the element. These box models are generated using Pupetteer's `.boxModel()`.
Format of each file is a dictionary where each key is a `axNode.backendDOMNodeId`:
```
{
    ...
    "4": {
        "content": [
        {
            "x": 300,
            "y": 267.796875
        },
        ...
        ],
        "padding": [
        {
            "x": 300,
            "y": 267.796875
        },
        ...
        ],
        "border": [
        {
            "x": 300,
            "y": 267.796875
        },
        ...
        ],
        "margin": [
        {
            "x": 300,
            "y": 267.796875
        },
        ...
        ],
        "width": 275,
        "height": 183
    },
    ...
}
```

**default_1920-1080-bb.json** is a 1920x1080 resolution screenshot with bounding boxes for each UI element. A bounding box is a minimal rectangle that entirely encloses the DOM element. These bounding boxes are generated using Pupetteer's `.boundingBox()`. **NOTE: This is what we want**
Format of each file is a dictionary where each key is a `axNode.backendDOMNodeId`:
```
{
    ...
    "4": {
        "x": 300,
        "y": 267.796875,
        "width": 275,
        "height": 183
    },
    ...
}
```

**default_1920-1080-axtree.json** is a 1920x1080 resolution screenshot with accessibility info for each UI element.

```
{
      "nodeId": "2581",
      "ignored": false,
      "role": {
        "type": "internalRole",
        "value": "RootWebArea"
      },
      "name": {
        "type": "computedString",
        "value": "QQ Dreamer – games online now",
        "sources": [
          {
            "type": "relatedElement",
            "attribute": "aria-labelledby"
          },
          {
            "type": "attribute",
            "attribute": "aria-label"
          },
          ,,,
        ]
      },
      "properties": [
        {
          "name": "focusable",
          "value": {
            "type": "booleanOrUndefined",
            "value": true
          }
        },
        }..
      ],
      "childIds": [
        "2582"
      ],
      "backendDOMNodeId": 3,
      "frameId": "88DC140DF3D3ADCC09F7D9AA09FFED6A"
    },

```

## Models

#### Detectron2

Run:

```bash
cd detectron2
python3 train.py
```

#### YOLOv8

Run:

```bash
cd YOLO
python3 train.py
```
Outputs with labels will be stored in `YOLO/runs/detect/predict`.

In order to visualize the bounding box predictions from the model on the validation set, run the python notebook `YOLO/visualize_results_yolov8.ipynb`.

#### YOLONAS

Run:

Use a preexisting experiment config in the `YOLONAS/experiment_configs` directory, or write your own. This will specify the datapath to use, hyperparameters to train the model with, etc.

```bash
cd YOLONAS
python3 train.py --experiment <experiment_config_name>
```

Example:
```bash
cd YOLONAS
python3 train.py --experiment yolo_nas_l_webui
```

Outputs with labels will be stored in `checkpoints/experiment_name/RUN_<date_time>`.

In order to visualize the bounding box predictions from the model on the validation set, run the python notebook `YOLONAS/visualize_results_yolonas.ipynb`.


#### SAM

TODO - @Avanika
