# Create Bounding Box Label Dataset

Goal: Create a dataset of natural language description labels for bounding boxes that can be used to benchmark grounding capabilities of different methods.

## How to run

The current dataset of bounding boxes and their labels can be found here: `/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels/bb_labels.csv`.

Because many of the webpages are quite long, we want to create a new dataset that points to cropped images at the resolution of a standard screenshot.

To do so, run: `python3 create_cropped_images.py`.

This will create a cropped version of the dataset at `/home/kopsahlong/eclair-agents/eclair/webpage_segmentation/datasets/bounding-box-labels`.