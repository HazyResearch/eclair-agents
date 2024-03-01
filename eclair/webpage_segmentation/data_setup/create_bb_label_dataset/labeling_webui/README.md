# Labeling Webapp
**Goal:** Enable faster labeling of bounded elements on screenshots with natural language descriptions.

## Usage Instructions

### To launch the app 
Run: `python3 app.py`

### To label
Enter in a description for the element with a red box around it. Example "Blue facebook icon", "Search bar at the top of the page", "Contact us section within the See more section", etc.

Press enter once the description is written in the text box to save it and load a new image.

If you would like to skip one of the elements, press the 'Skip' button.

## Where to find the saved data
Check the file path: `/eclair-agents/eclair/webpage_segmentation/datasets/bb_labels.csv`for saved data.


## TODO
* had to download the images -- all to my computer first and then unzip on my computer, then over to server
* had to symlink the folder with these images into the labeling webapp