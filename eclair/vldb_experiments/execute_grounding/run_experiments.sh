#!/bin/bash

python main.py --inference_model "CogAgent" --bbox_generator none --dataset "WebUI"
python main.py --inference_model "CogAgent" --bbox_generator none --dataset "Mind2Web"

python main.py --inference_model "CogAgent" --bbox_generator none --dataset "Mind2Web" --quant 4
python main.py --inference_model "CogAgent" --bbox_generator none --dataset "WebUI" --quant 4

python main.py --inference_model "GPT4V" --bbox_generator "yolo_nas_l_webui" --regenerate_boxes True --prompt_strat "set-of-marks" --dataset "Mind2Web"
python main.py --inference_model "GPT4V" --bbox_generator "ground_truth" --regenerate_boxes True --prompt_strat "set-of-marks" --dataset "Mind2Web"

python main.py --inference_model "GPT4V" --prompt_strat "axtree" --input_image True --dataset "WebUI"
python main.py --inference_model "GPT4V" --prompt_strat "axtree" --input_image False --dataset "WebUI"

python main.py --inference_model "GPT4V" --bbox_generator "yolo_nas_l_webui" --regenerate_boxes True --prompt_strat "set-of-marks" --dataset "WebUI"
python main.py --inference_model "GPT4V" --bbox_generator "ground_truth" --regenerate_boxes True --prompt_strat "set-of-marks" --dataset "WebUI"
