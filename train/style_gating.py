import torch
import torch.nn as nn
import clip
from typing import Optional, Tuple


class StyleGatingModule(nn.Module):
    """
    Style gating module that uses CLIP image embeddings to dynamically blend
    between two LoRA adapters using sigmoid-gated interpolation.
    """
    def __init__(
        self, 
        clip_model_name: str = "ViT-B/32",
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_styles: int = 2
    ):
        super().__init__()
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Style gating network
        self.style_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_styles - 1),  # n-1 gates for n styles
            nn.Sigmoid()
        )
        
        self.num_styles = num_styles
        
    def extract_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP image features from input images."""
        with torch.no_grad():
            # Ensure images are in the right format for CLIP
            if images.dim() == 4:  # Batch of images
                batch_size = images.size(0)
                clip_features = []
                
                for i in range(batch_size):
                    # Convert from [-1, 1] to [0, 1] if needed
                    img = (images[i] + 1) / 2 if images[i].min() < 0 else images[i]
                    
                    # CLIP expects PIL or specific tensor format
                    img = torch.clamp(img, 0, 1)
                    features = self.clip_model.encode_image(img.unsqueeze(0))
                    clip_features.append(features)
                
                clip_features = torch.cat(clip_features, dim=0)
            else:
                # Single image
                img = (images + 1) / 2 if images.min() < 0 else images
                img = torch.clamp(img, 0, 1)
                clip_features = self.clip_model.encode_image(img.unsqueeze(0))
                
        return clip_features.float()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute style gating weights based on image content.
        
        Args:
            images: Input images tensor [B, C, H, W]
            
        Returns:
            Style weights tensor [B, num_styles] with values summing to 1
        """
        # Extract CLIP features
        clip_features = self.extract_clip_features(images)
        
        # Compute gating weights
        gates = self.style_gate(clip_features)  # [B, num_styles-1]
        
        # Convert sigmoid outputs to proper probability distribution
        if self.num_styles == 2:
            # For 2 styles: w1 = gate, w2 = 1 - gate
            weights = torch.stack([gates.squeeze(-1), 1 - gates.squeeze(-1)], dim=-1)
        else:
            # For n > 2 styles: use stick-breaking construction
            weights = self._stick_breaking(gates)
            
        return weights
    
    def _stick_breaking(self, gates: torch.Tensor) -> torch.Tensor:
        """Convert sigmoid gates to probability distribution using stick-breaking."""
        batch_size = gates.size(0)
        weights = torch.zeros(batch_size, self.num_styles, device=gates.device)
        
        remaining = torch.ones(batch_size, device=gates.device)
        
        for i in range(self.num_styles - 1):
            weights[:, i] = gates[:, i] * remaining
            remaining = remaining - weights[:, i]
            
        weights[:, -1] = remaining  # Last weight gets remaining probability
        
        return weights