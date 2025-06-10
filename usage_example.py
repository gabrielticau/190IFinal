#!/usr/bin/env python3
"""
Usage Example for Multi-Style LoRA with CLIP-based Style Gating

This example demonstrates how to:
1. Train a multi-style LoRA model with dynamic style blending
2. Use the trained model for style transfer with automatic style detection
3. Create style comparison visualizations
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Example training configuration
TRAINING_CONFIG = {
    'pretrained_path': 'runwayml/stable-diffusion-v1-5',
    'style1_path': './datasets/picasso',
    'style1_prefix': 'A Picasso painting, ',
    'style2_path': './datasets/van_gogh', 
    'style2_prefix': 'A Van Gogh painting, ',
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'train_strength': 0.3,
    'learning_rate': 1e-5,
    'gating_lr': 1e-4,
    'batch_size': 4,
    'max_epochs': 100,
    'style_consistency_weight': 0.1,
    'diversity_weight': 0.05,
    'save_name': 'picasso_van_gogh_blend'
}

def train_multi_style_model():
    """Example training command"""
    cmd = f"""
    python multi_style_trainer.py \\
        --pretrained_path {TRAINING_CONFIG['pretrained_path']} \\
        --style1_path {TRAINING_CONFIG['style1_path']} \\
        --style1_prefix "{TRAINING_CONFIG['style1_prefix']}" \\
        --style2_path {TRAINING_CONFIG['style2_path']} \\
        --style2_prefix "{TRAINING_CONFIG['style2_prefix']}" \\
        --lora_r {TRAINING_CONFIG['lora_r']} \\
        --lora_alpha {TRAINING_CONFIG['lora_alpha']} \\
        --lora_dropout {TRAINING_CONFIG['lora_dropout']} \\
        --train_strength {TRAINING_CONFIG['train_strength']} \\
        --learning_rate {TRAINING_CONFIG['learning_rate']} \\
        --gating_lr {TRAINING_CONFIG['gating_lr']} \\
        --batch_size {TRAINING_CONFIG['batch_size']} \\
        --max_epochs {TRAINING_CONFIG['max_epochs']} \\
        --style_consistency_weight {TRAINING_CONFIG['style_consistency_weight']} \\
        --diversity_weight {TRAINING_CONFIG['diversity_weight']} \\
        --save_name {TRAINING_CONFIG['save_name']}
    """
    print("Training command:")
    print(cmd)
    return cmd

def inference_example():
    """Example inference command"""
    cmd = f"""
    python style_transfer_gated.py \\
        --pretrained_path {TRAINING_CONFIG['pretrained_path']} \\
        --gated_model_path ./multi_style_final/{TRAINING_CONFIG['save_name']} \\
        --image_path ./test_images/portrait.jpg \\
        --prompt "portrait of a woman" \\
        --strength 0.7 \\
        --num_inference_steps 20 \\
        --create_comparison \\
        --output_path styled_portrait.jpg \\
        --comparison_path portrait_comparison.png
    """
    print("Inference command:")
    print(cmd)
    return cmd

def analyze_style_weights():
    """Example of analyzing style weights for different image types"""
    
    # This would be used after training
    from train.style_gating import StyleGatingModule
    
    # Load trained style gating module
    style_gating = StyleGatingModule(num_styles=2)
    style_gating.load_state_dict(
        torch.load(f'./multi_style_final/{TRAINING_CONFIG["save_name"]}/style_gating.pt')
    )
    style_gating.eval()
    
    # Test images of different types
    test_images = [
        ('portrait.jpg', 'Human portrait'),
        ('landscape.jpg', 'Natural landscape'), 
        ('still_life.jpg', 'Still life objects'),
        ('abstract.jpg', 'Abstract composition')
    ]
    
    results = []
    
    for img_path, description in test_images:
        if os.path.exists(img_path):
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            
            # Transform for model input
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            img_tensor = transform(image).unsqueeze(0)
            
            # Get style weights
            with torch.no_grad():
                weights = style_gating(img_tensor)
            
            style1_weight = weights[0, 0].item()
            style2_weight = weights[0, 1].item()
            
            results.append({
                'image': description,
                'style1_weight': style1_weight,
                'style2_weight': style2_weight,
                'dominant_style': 'Style 1' if style1_weight > style2_weight else 'Style 2'
            })
    
    # Display results
    print("\nStyle Weight Analysis:")
    print("-" * 60)
    for result in results:
        print(f"{result['image']:20} | Style 1: {result['style1_weight']:.3f} | "
              f"Style 2: {result['style2_weight']:.3f} | Dominant: {result['dominant_style']}")
    
    return results

def create_advanced_blend_example():
    """Example of creating custom style blends"""
    
    blend_examples = [
        {
            'name': 'Geometric Portrait',
            'prompt': 'geometric portrait of a person',
            'manual_blend': (0.8, 0.2),  # More Picasso-like
            'strength': 0.6
        },
        {
            'name': 'Expressive Landscape', 
            'prompt': 'expressive landscape with swirling clouds',
            'manual_blend': (0.2, 0.8),  # More Van Gogh-like
            'strength': 0.7
        },
        {
            'name': 'Balanced Still Life',
            'prompt': 'still life with fruits and flowers',
            'manual_blend': (0.5, 0.5),  # Equal blend
            'strength': 0.65
        }
    ]
    
    print("\nAdvanced Blend Examples:")
    print("-" * 50)
    
    for example in blend_examples:
        cmd = f"""
        python style_transfer_gated.py \\
            --pretrained_path {TRAINING_CONFIG['pretrained_path']} \\
            --gated_model_path ./multi_style_final/{TRAINING_CONFIG['save_name']} \\
            --image_path ./input_images/test.jpg \\
            --prompt "{example['prompt']}" \\
            --manual_blend "{example['manual_blend'][0]},{example['manual_blend'][1]}" \\
            --strength {example['strength']} \\
            --output_path {example['name'].lower().replace(' ', '_')}.jpg
        """
        
        print(f"\n{example['name']}:")
        print(f"  Blend: {example['manual_blend']}")
        print(f"  Command: {cmd.strip()}")

def performance_tips():
    """Performance optimization tips"""
    
    tips = [
        "Use bf16 precision for faster training and lower memory usage",
        "Increase batch size if you have sufficient GPU memory",
        "Use gradient accumulation for effective larger batch sizes",
        "Monitor style weight distributions to ensure balanced learning",
        "Adjust style consistency weight if styles are too mixed",
        "Use different LoRA ranks for different complexity requirements",
        "Consider using gradient checkpointing for memory efficiency",
        "Precompute CLIP features for faster training iterations"
    ]
    
    print("\nPerformance Tips:")
    print("-" * 30)
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")

def troubleshooting_guide():
    """Common issues and solutions"""
    
    issues = [
        {
            'problem': 'Style gating always favors one style',
            'solution': 'Increase diversity_weight or decrease style_consistency_weight'
        },
        {
            'problem': 'Generated images look too mixed/unclear',
            'solution': 'Increase style_consistency_weight to encourage cleaner style separation'
        },
        {
            'problem': 'CLIP features not working properly',
            'solution': 'Ensure images are properly normalized and in RGB format'
        },
        {
            'problem': 'Training is unstable',
            'solution': 'Reduce learning rates, add gradient clipping, or use smaller batch sizes'
        },
        {
            'problem': 'Style transfer results are poor',
            'solution': 'Check if LoRA adapters are properly loaded and weights are reasonable'
        }
    ]
    
    print("\nTroubleshooting Guide:")
    print("-" * 40)
    for issue in issues:
        print(f"Problem: {issue['problem']}")
        print(f"Solution: {issue['solution']}\n")

if __name__ == "__main__":
    print("=== Multi-Style LoRA with CLIP-based Style Gating ===\n")
    
    print("1. Training Configuration:")
    print("-" * 30)
    for key, value in TRAINING_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n2. Training Command:")
    train_multi_style_model()
    
    print("\n3. Inference Example:")
    inference_example()
    
    print("\n4. Advanced Blend Examples:")
    create_advanced_blend_example()
    
    performance_tips()
    troubleshooting_guide()
    
    print("\n=== Setup Complete ===")
    print("You can now train your multi-style LoRA model and perform dynamic style blending!")