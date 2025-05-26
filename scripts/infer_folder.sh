#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

OUT_FOLDER="out"

if [ ! -d "$OUT_FOLDER" ]; then
    mkdir -p "$OUT_FOLDER"
fi

python inference/transfer_style_folder.py \
--pretrained_path "/home/trapoom555/data/hf_models/stable-diffusion-v1-5" \
--lora_path "/home/trapoom555/code/Diffusion-LoRA-Style-Transfer/lora_ckpt/monet_all" \
--ip_adapter_path "/home/trapoom555/data/hf_models/IP-Adapter" \
--folder_path "/home/trapoom555/data/monet_data/photo_jpg/" \
--image_cond_scale 0.3 \
--strength 0.5 \
--prompt "A Monet painting" \
--infer_steps 100 \
--save_path "$OUT_FOLDER"