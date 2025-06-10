#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python inference/transfer_style_double_lora.py \
--pretrained_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
--lora_path_1 "/home/gabit/190IFinal/lora_ckpt/monet_all" \
--lora_path_2 "/home/gabit/190IFinal/lora_ckpt/ukiyoe_all" \
--lora_weight_1 0.0 \
--lora_weight_2 1.0 \
--ip_adapter_path "h94/IP-Adapter" \
--image_path "/home/gabit/190IFinal/landscape.jpg" \
--image_cond_scale 0.3 \
--strength 0.5 \
--prompt "A painting" \
--infer_steps 100 \
--save_path "styled_night_merged_lora0:1.jpg"
