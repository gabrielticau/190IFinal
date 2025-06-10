import torch
import lightning.pytorch as pl
from diffusers import StableDiffusionPipeline


class DiffusionLoRAModule(pl.LightningModule):
    def __init__(self, pipeline: StableDiffusionPipeline, lr: float, train_strength: float = 0.3):
        super().__init__()
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.text_encoder = pipeline.text_encoder
        self.train_strength = train_strength
        self.lr = lr

        self.prefix = 'A painting, '
    
    def training_step(self, batch, batch_idx):
        images, captions = batch
        B = images.size(0)

        captions = [self.prefix + c for c in captions]
        tokens = self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True).input_ids.to(self.device)

        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, int(1000*self.train_strength), (B,), device=self.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=self.text_encoder(tokens)[0])['sample']

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.lr)
        return optimizer