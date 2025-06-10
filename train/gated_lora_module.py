import torch
import torch.nn as nn
import lightning.pytorch as pl
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Dict, Any, Optional
import copy


class GatedLoRAModule(pl.LightningModule):
    """
    Multi-style LoRA module with CLIP-based style gating for dynamic style blending.
    """
    def __init__(
        self, 
        pipeline: StableDiffusionPipeline, 
        style_prefixes: List[str],
        lora_configs: List[Dict[str, Any]],
        lr: float = 1e-5,
        train_strength: float = 0.3,
        gating_lr: float = 1e-4,
        style_consistency_weight: float = 0.1,
        diversity_weight: float = 0.05
    ):
        super().__init__()
        
        # Enable manual optimization for multiple optimizers
        self.automatic_optimization = False
        
        # Store pipeline components
        self.base_unet = pipeline.unet
        self.vae = pipeline.vae
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.text_encoder = pipeline.text_encoder
        
        # Training parameters
        self.train_strength = train_strength
        self.lr = lr
        self.gating_lr = gating_lr
        self.style_consistency_weight = style_consistency_weight
        self.diversity_weight = diversity_weight
        
        # Style information
        self.style_prefixes = style_prefixes
        self.num_styles = len(style_prefixes)
        
        # Create multiple LoRA adapters
        self.lora_adapters = nn.ModuleList()
        for i, lora_config in enumerate(lora_configs):
            # Create a copy of the base UNet for each style
            unet_copy = copy.deepcopy(self.base_unet)
            lora_adapter = get_peft_model(unet_copy, LoraConfig(**lora_config))
            self.lora_adapters.append(lora_adapter)
        
        # Style gating module
        from train.style_gating import StyleGatingModule
        self.style_gating = StyleGatingModule(num_styles=self.num_styles)
        
        # Cache for style weights (for logging)
        self.last_style_weights = None
        
    def blend_lora_outputs(
        self, 
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        style_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Blend outputs from multiple LoRA adapters based on style weights.
        """
        batch_size = noisy_latents.size(0)
        blended_output = torch.zeros_like(noisy_latents)
        
        for i, lora_adapter in enumerate(self.lora_adapters):
            # Get prediction from this LoRA adapter
            noise_pred = lora_adapter(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states=encoder_hidden_states
            )['sample']
            
            # Weight by style gating
            style_weight = style_weights[:, i].view(-1, 1, 1, 1)
            blended_output += style_weight * noise_pred
            
        return blended_output
    
    def training_step(self, batch, batch_idx):
        # Get optimizers
        opt_lora, opt_gating = self.optimizers()
        
        images, captions = batch
        batch_size = images.size(0)
        
        # Zero gradients
        opt_lora.zero_grad()
        opt_gating.zero_grad()
        
        # Compute style weights based on image content
        style_weights = self.style_gating(images)
        self.last_style_weights = style_weights.detach()
        
        # Prepare captions with style-specific prefixes
        # Use the dominant style for each image
        dominant_styles = torch.argmax(style_weights, dim=1)
        prefixed_captions = []
        for i, caption in enumerate(captions):
            style_idx = dominant_styles[i].item()
            prefixed_captions.append(self.style_prefixes[style_idx] + caption)
        
        # Tokenize captions
        tokens = self.tokenizer(
            prefixed_captions, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).input_ids.to(self.device)
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        
        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, int(1000 * self.train_strength), 
            (batch_size,), 
            device=self.device, 
            dtype=torch.long
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(tokens)[0]
        
        # Blend LoRA outputs
        noise_pred = self.blend_lora_outputs(
            noisy_latents, timesteps, encoder_hidden_states, style_weights
        )
        
        # Main denoising loss
        main_loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Style consistency loss - encourage consistent style usage
        style_entropy = -torch.sum(style_weights * torch.log(style_weights + 1e-8), dim=1)
        consistency_loss = torch.mean(style_entropy)
        
        # Diversity loss - encourage different styles to be used
        mean_style_weights = torch.mean(style_weights, dim=0)
        diversity_loss = -torch.sum(mean_style_weights * torch.log(mean_style_weights + 1e-8))
        
        # Total loss
        total_loss = (
            main_loss + 
            self.style_consistency_weight * consistency_loss -
            self.diversity_weight * diversity_loss
        )
        
        # Manual backward pass
        self.manual_backward(total_loss)
        
        # Step optimizers
        opt_lora.step()
        opt_gating.step()
        
        # Logging
        self.log('train_loss', main_loss, prog_bar=True)
        self.log('consistency_loss', consistency_loss)
        self.log('diversity_loss', diversity_loss)
        self.log('total_loss', total_loss)
        
        # Log style usage statistics
        for i in range(self.num_styles):
            self.log(f'style_{i}_usage', torch.mean(style_weights[:, i]))
        
        return total_loss
    
    def configure_optimizers(self):
        # Separate optimizers for LoRA adapters and style gating
        lora_params = []
        for adapter in self.lora_adapters:
            lora_params.extend(adapter.parameters())
        
        optimizer_lora = torch.optim.AdamW(lora_params, lr=self.lr)
        optimizer_gating = torch.optim.AdamW(
            self.style_gating.parameters(), 
            lr=self.gating_lr
        )
        
        return [optimizer_lora, optimizer_gating]
    
    def save_style_adapters(self, save_dir: str):
        """Save individual LoRA adapters."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i, adapter in enumerate(self.lora_adapters):
            adapter_path = os.path.join(save_dir, f'style_{i}_lora')
            adapter.save_pretrained(adapter_path)
            
        # Save style gating module
        gating_path = os.path.join(save_dir, 'style_gating.pt')
        torch.save(self.style_gating.state_dict(), gating_path)
        
        print(f"Saved {self.num_styles} LoRA adapters and style gating module to {save_dir}")
        
    def load_style_adapters(self, save_dir: str):
        """Load individual LoRA adapters."""
        import os
        
        for i in range(self.num_styles):
            adapter_path = os.path.join(save_dir, f'style_{i}_lora')
            if os.path.exists(adapter_path):
                self.lora_adapters[i] = PeftModel.from_pretrained(
                    self.lora_adapters[i], adapter_path
                )
                
        # Load style gating module
        gating_path = os.path.join(save_dir, 'style_gating.pt')
        if os.path.exists(gating_path):
            self.style_gating.load_state_dict(torch.load(gating_path))
            
        print(f"Loaded {self.num_styles} LoRA adapters and style gating module from {save_dir}")
        
    def get_style_weights_for_image(self, image: torch.Tensor) -> torch.Tensor:
        """Get style weights for a single image."""
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return self.style_gating(image)