#!/bin/bash

export PYTHONPATH="$(pwd)"

# Set your dataset paths here
MONET_PATH="C:\\Users\\jenni\\190IFinal\\data\\images\\photo1"
UKIYO_E_PATH="C:\\Users\\jenni\\190IFinal\\data\\images\\photo2"

# Create JSON string with proper escaping
python train/trainer.py \
  --pretrained_path "stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --monet_path "/c/Users/jenni/190IFinal/data/images/photo1_short" \
  --ukiyo_e_path "/c/Users/jenni/190IFinal/data/images/photo2_short" \
  --center_crop false \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --train_strength 0.8 \
  --learning_rate 3e-6 \
  --batch_size 3 \
  --grad_accumulation 3 \
  --max_epochs 1 \
  --multi_style_prob 0.4 \
  --save_name "monet_ukiyo_fusion"

# Completion message
echo "Training complete! Models saved as monet_all and monet_ukiyo_fusion"
echo "Style tokens available: <monet> and <ukiyo_e>"