# Execute: Grounding Experiment

Ideal interface:

```bash
python main.py --inference_model "GPT4" --bbox_generator "YOLONAS" --prompt_strat "set-of-marks" --dataset "mind2web"
python main.py --inference_model "GPT4" --bbox_generator "YOLONAS" --prompt_strat "json_string" --dataset "webui"

python main.py --inference_model "GPT4" --bbox_generator "ground_truth" --prompt_strat "json_string" --dataset "mind2web"
python main.py --inference_model "GPT4" --bbox_generator "WebUI" --prompt_strat "json_string" --dataset "mind2web"

python main.py --inference_model "CogAgent" --bbox_generator "WebUI" --prompt_strat "json_string" --dataset "mind2web"
python main.py --inference_model "CogAgent" --bbox_generator None --dataset "mind2web"
```

## Datasets

All with 40/40/40 split between buttons/links/textfields.

* Mind2Web
* WebUI
* Desktop [optional]

## Ablations

* YOLONAS bboxes + Image => GPT-4V (Set-of-marks):  Pass image into YOLONAS, get a set of bboxes out. Overlay those bboxes onto the image using set-of-marks. Feed set-of-marks image into GPT-4V with action suggestion, get out element ID
* YOLONAS bboxes + Image => GPT-4V (JSON string):  Pass image into YOLONAS, get a set of bboxes out. Convert those bboxes into a JSON string, feed that as text along with the original raw image to GPT-4V, get it to output the ID of the element.
* WebUI bboxes + Image => GPT-4V (Set-of-marks):  Pass image into WebUI Model, get a set of bboxes out. Overlay those bboxes onto the image using set-of-marks. Feed set-of-marks image into GPT-4V with action suggestion, get out element ID
* WebUI bboxes + Image => GPT-4V (JSON string):  Pass image into WebUI Model, get a set of bboxes out. Convert those bboxes into a JSON string, feed that as text along with the original raw image to GPT-4V, get it to output the ID of the element.
* Ground-truth bboxes + Image => GPT-4V (set-of-marks): Take the known gt bboxes of image. Overlay those bboxes onto the image using set-of-marks. Feed set-of-marks image into GPT-4V with action suggestion, get out element ID  \todo{Unclear if possible with Mind2Web b/c they don't preserve good bboxes for elements}
* Ground-truth bboxes + Image => GPT-4V (JSON string): Take the known gt bboxes of image. Convert those bboxes into a JSON string, feed that as text along with the original raw image to GPT-4V, get it to output the ID of the element.\todo{Unclear if possible with Mind2Web b/c they don't preserve good bboxes for elements}
* Image => CogAgent: Take the raw image, feed it into CogAgent. Also feed in the action suggestion. Get bbox as output. * Ground-truth bboxes + * Image => CogAgent:  Take the known gt bboxes of image. Convert those bboxes into a JSON string, feed that as text along with the raw image into CogAgent. Get bbox as output.  \todo{Unclear if possible with Mind2Web b/c they don't preserve good bboxes for elements}
