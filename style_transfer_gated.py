import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoPipelineForImage2Image, StableDiffusionPipeline
import matplotlib.pyplot as plt

from train.gated_lora_module import GatedLoRAModule
from train.style_gating import StyleGatingModule


class GatedStyleTransfer:
    """
    Style transfer using gated LoRA adapters with dynamic style blending.
    """
    def __init__(
        self, 
        pretrained_path: str,
        gated_model_path: str,
        ip_adapter_path: str = None
    ):
        # Load base pipeline
        self.pipeline = AutoPipelineForImage2Image.from_pretrained(pretrained_path)
        self.pipeline.to('cuda')
        
        # Load IP-Adapter if provided
        if ip_adapter_path:
            self.pipeline.load_ip_adapter(
                ip_adapter_path, 
                subfolder='models', 
                weight_name='ip-adapter_sd15.bin'
            )
            self.has_ip_adapter = True
        else:
            self.has_ip_adapter = False
        
        # Load gated LoRA model
        self.gated_model = self._load_gated_model(gated_model_path)
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        if self.has_ip_adapter:
            self.ip_transforms = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
    
    def _load_gated_model(self, model_path: str):
        """Load the trained gated LoRA model."""
        # This is a simplified version - in practice, you'd need to 
        # properly reconstruct the model with the same parameters
        # For now, we'll create a mock structure
        
        # Load style gating module
        style_gating = StyleGatingModule(num_styles=2)
        gating_weights_path = f"{model_path}/style_gating.pt"
        try:
            style_gating.load_state_dict(torch.load(gating_weights_path))
            style_gating.eval()
            return style_gating
        except FileNotFoundError:
            print(f"Warning: Could not load style gating weights from {gating_weights_path}")
            return style_gating
    
    def get_style_weights(self, image: Image.Image) -> torch.Tensor:
        """Get style blending weights for an input image."""
        # Transform image
        img_tensor = self.image_transforms(image).unsqueeze(0).cuda()
        
        # Get style weights
        with torch.no_grad():
            weights = self.gated_model(img_tensor)
        
        return weights.cpu()
    
    def transfer_style(
        self,
        image: Image.Image,
        prompt: str,
        style_weights: torch.Tensor = None,
        strength: float = 0.7,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        image_cond_scale: float = 1.0,
        manual_style_blend: tuple = None
    ) -> Image.Image:
        """
        Perform style transfer with dynamic or manual style blending.
        
        Args:
            image: Input image
            prompt: Text prompt
            style_weights: Pre-computed style weights (optional)
            strength: Denoising strength
            num_inference_steps: Number of inference steps
            guidance_scale: CFG guidance scale
            image_cond_scale: IP-Adapter conditioning scale
            manual_style_blend: Manual blend weights (style1_weight, style2_weight)
        """
        # Get style weights if not provided
        if style_weights is None and manual_style_blend is None:
            style_weights = self.get_style_weights(image)
        elif manual_style_blend is not None:
            # Use manual blending
            w1, w2 = manual_style_blend
            total = w1 + w2
            style_weights = torch.tensor([[w1/total, w2/total]])
        
        print(f"Style weights: {style_weights[0].numpy()}")
        
        # Transform image
        img_tensor = self.image_transforms(image)
        
        # Prepare IP-Adapter image if available
        ip_image = None
        if self.has_ip_adapter:
            ip_image = self.ip_transforms(image).unsqueeze(0)
            self.pipeline.set_ip_adapter_scale(image_cond_scale)
        
        # Create style-weighted prompt
        style1_weight, style2_weight = style_weights[0]
        
        # For this example, assume we have Picasso and Van Gogh styles
        if style1_weight > style2_weight:
            dominant_style = "Picasso"
            style_prompt = f"A Picasso painting, {prompt}"
        else:
            dominant_style = "Van Gogh"
            style_prompt = f"A Van Gogh painting, {prompt}"
        
        print(f"Dominant style: {dominant_style}")
        print(f"Style prompt: {style_prompt}")
        
        # Generate image
        kwargs = {
            'prompt': style_prompt,
            'image': img_tensor,
            'strength': strength,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
        }
        
        if self.has_ip_adapter and ip_image is not None:
            kwargs['ip_adapter_image'] = ip_image
        
        result = self.pipeline(**kwargs)
        
        return result.images[0], style_weights
    
    def create_style_comparison(
        self,
        image: Image.Image,
        prompt: str,
        save_path: str = None
    ):
        """Create a comparison showing different style blends."""
        
        # Different manual blends to try
        blend_configs = [
            (1.0, 0.0, "Pure Style 1"),
            (0.7, 0.3, "Style 1 Dominant"),
            (0.5, 0.5, "Equal Blend"),
            (0.3, 0.7, "Style 2 Dominant"),
            (0.0, 1.0, "Pure Style 2"),
        ]
        
        # Get automatic style weights
        auto_weights = self.get_style_weights(image)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Style Transfer Comparison\nPrompt: "{prompt}"', fontsize=14)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Auto-weighted result
        auto_result, _ = self.transfer_style(
            image, prompt, style_weights=auto_weights
        )
        axes[0, 1].imshow(auto_result)
        axes[0, 1].set_title(f'Auto Blend\n({auto_weights[0, 0]:.2f}, {auto_weights[0, 1]:.2f})')
        axes[0, 1].axis('off')
        
        # Manual blends
        for i, (w1, w2, title) in enumerate(blend_configs[:4]):
            row = (i + 2) // 3
            col = (i + 2) % 3
            
            result, _ = self.transfer_style(
                image, prompt, manual_style_blend=(w1, w2)
            )
            axes[row, col].imshow(result)
            axes[row, col].set_title(f'{title}\n({w1}, {w2})')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Style transfer with gated LoRA')
    
    # Model paths
    parser.add_argument('--pretrained_path', type=str, required=True,
                       help='Path to pretrained Stable Diffusion model')
    parser.add_argument('--gated_model_path', type=str, required=True,
                       help='Path to trained gated LoRA model')
    parser.add_argument('--ip_adapter_path', type=str, default=None,
                       help='Path to IP-Adapter model (optional)')
    
    # Input parameters
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt for style transfer')
    
    # Generation parameters
    parser.add_argument('--strength', type=float, default=0.7,
                       help='Denoising strength')
    parser.add_argument('--num_inference_steps', type=int, default=20,
                       help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='CFG guidance scale')
    parser.add_argument('--image_cond_scale', type=float, default=1.0,
                       help='IP-Adapter conditioning scale')
    
    # Style blending
    parser.add_argument('--manual_blend', type=str, default=None,
                       help='Manual style blend as "w1,w2" (e.g., "0.7,0.3")')
    parser.add_argument('--create_comparison', action='store_true',
                       help='Create style comparison grid')
    
    # Output
    parser.add_argument('--output_path', type=str, default='styled_output.jpg',
                       help='Output image path')
    parser.add_argument('--comparison_path', type=str, default='style_comparison.png',
                       help='Comparison grid output path')
    
    args = parser.parse_args()
    
    # Initialize style transfer
    print("Loading models...")
    style_transfer = GatedStyleTransfer(
        pretrained_path=args.pretrained_path,
        gated_model_path=args.gated_model_path,
        ip_adapter_path=args.ip_adapter_path
    )
    
    # Load input image
    print(f"Loading image: {args.image_path}")
    input_image = Image.open(args.image_path).convert('RGB')
    
    if args.create_comparison:
        # Create comparison grid
        print("Creating style comparison...")
        style_transfer.create_style_comparison(
            input_image, 
            args.prompt, 
            args.comparison_path
        )
    else:
        # Single style transfer
        manual_blend = None
        if args.manual_blend:
            w1, w2 = map(float, args.manual_blend.split(','))
            manual_blend = (w1, w2)
        
        print("Performing style transfer...")
        result_image, style_weights = style_transfer.transfer_style(
            image=input_image,
            prompt=args.prompt,
            manual_style_blend=manual_blend,
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_cond_scale=args.image_cond_scale
        )
        
        # Save result
        result_image.save(args.output_path)
        print(f"Result saved to: {args.output_path}")
        print(f"Style weights used: {style_weights[0].numpy()}")


if __name__ == "__main__":
    main()