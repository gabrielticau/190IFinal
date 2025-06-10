#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python inference/transfer_style.py \
--pretrained_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
--lora_path "/home/gabit/190IFinal/lora_ckpt/monet+ukiyoe_dataset_all" \
--ip_adapter_path "h94/IP-Adapter" \
--image_path "/home/gabit/190IFinal/house.jpg" \
--image_cond_scale 0.3 \
--strength 0.5 \
--prompt "A Monet Painting" \
--infer_steps 100 \
--save_path "styled_image_mixed_dataset.jpg"