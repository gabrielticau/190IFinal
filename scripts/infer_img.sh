#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

python inference/transfer_style.py \
--pretrained_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
--lora_path "/root/190IFinal/lora_ckpt/monet_all" \
--ip_adapter_path "h94/IP-Adapter" \
--image_path "/root/190IFinal/Water-lily-pond.jpg" \
--image_cond_scale 0.65 \
--strength 0.8 \
--prompt "A Monet painting" \
--infer_steps 100 \
--save_path "styled_imageHighStrength.jpg"