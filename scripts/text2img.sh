#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python inference/text2img.py \
--prompt "A Monet painting" \
--pretrained_path "/home/trapoom555/data/hf_models/stable-diffusion-v1-5" \
--lora_path "/home/trapoom555/code/Diffusion-LoRA-Style-Transfer/lora_ckpt/monet_all" \
--infer_steps 100 \
--save_path "output.jpg"