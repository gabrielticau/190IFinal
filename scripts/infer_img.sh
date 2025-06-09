#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"

python inference/transfer_style.py \
--pretrained_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
--lora_path "lora_ckpt/monet_ukiyo_fusion" \
--ip_adapter_path "h94/IP-Adapter" \
--image_path "C:/Users/jenni/190IFinal/house (1).jpg" \
--image_cond_scale 0.65 \
--strength 0.8 \
--prompt "A peaceful landscape in <monet> style" \
--infer_steps 100 \
--save_path "styled_image_JapaneseBias1.jpg"
