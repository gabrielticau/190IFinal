import torch
import lightning.pytorch as pl
from diffusers import StableDiffusionPipeline
import random

class DiffusionLoRAModule(pl.LightningModule):
    def __init__(self, pipeline: StableDiffusionPipeline, lr: float, train_strength: float = 0.3, 
                 style_tokens: list = None, multi_style_prob: float = 0.3):
        super().__init__()
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.text_encoder = pipeline.text_encoder
        self.train_strength = train_strength
        self.lr = lr
        self.multi_style_prob = multi_style_prob

        self.style_tokens = style_tokens or ['<monet>', '<ukiyo_e>']
        special_tokens_dict = {'additional_special_tokens': self.style_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        self._initialize_style_embeddings()
        self._freeze_text_encoder()

        self.style_templates = {
            'monet': [
                "An impressionist painting of {caption}",
                "A {caption} in the style of Claude Monet",
                "Impressionist artwork depicting {caption}",
                "{caption} painted with soft brushstrokes and natural light"
            ],
            'ukiyo_e': [
                "A Japanese woodblock print of {caption}",
                "Traditional ukiyo-e artwork showing {caption}",
                "Japanese art depicting {caption}",
                "{caption} in the style of traditional Japanese prints"
            ],
            'fusion': [
                "A {caption} combining impressionist and Japanese artistic styles",
                "An artistic fusion of Western impressionism and Japanese ukiyo-e showing {caption}",
                "{caption} painted in a style blending Monet's impressionism with Japanese woodblock techniques"
            ]
        }

    def _initialize_style_embeddings(self):
        art_concepts = {
            '<monet>': ['impressionist', 'water', 'light', 'garden', 'brushstroke', 'nature'],
            '<ukiyo_e>': ['japanese', 'woodblock', 'traditional', 'elegant', 'delicate', 'seasonal']
        }

        with torch.no_grad():
            for token in self.style_tokens:
                if token in art_concepts:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    concept_words = art_concepts[token]
                    concept_embeddings = []
                    for word in concept_words:
                        word_ids = self.tokenizer.encode(word, add_special_tokens=False)
                        if word_ids:
                            word_embed = self.text_encoder.text_model.embeddings.token_embedding(
                                torch.tensor(word_ids[0]).to(self.device)
                            )
                            concept_embeddings.append(word_embed)
                    if concept_embeddings:
                        avg_embedding = torch.stack(concept_embeddings).mean(dim=0)
                        noise = torch.randn_like(avg_embedding) * 0.02
                        avg_embedding = avg_embedding + noise
                        self.text_encoder.text_model.embeddings.token_embedding.weight[token_id] = avg_embedding
                        print(f"Initialized {token} embedding from {len(concept_embeddings)} concept words")

    def _freeze_text_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for token in self.style_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id < len(self.text_encoder.text_model.embeddings.token_embedding.weight):
                self.text_encoder.text_model.embeddings.token_embedding.weight[token_id].requires_grad = True
                print(f"Enabled training for {token} (ID: {token_id})")

    def _generate_styled_prompt(self, caption: str, style):
        if isinstance(style, list) and len(style) > 1:
            if random.random() < 0.5:
                template = random.choice(self.style_templates['fusion'])
                return template.format(caption=caption)
            else:
                style_tokens_str = ' and '.join([f'<{s}>' for s in style])
                return f"A {caption} in the style of {style_tokens_str}"
        else:
            style_name = style if isinstance(style, str) else style[0]
            if random.random() < 0.3 and style_name in self.style_templates:
                template = random.choice(self.style_templates[style_name])
                return template.format(caption=caption)
            return f"A {caption} in the style of <{style_name}>"

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, captions, style_labels = batch
        else:
            images, captions = batch
            style_labels = ['monet'] * len(captions)

        B = images.size(0)
        styled_captions = [self._generate_styled_prompt(c, s) for c, s in zip(captions, style_labels)]

        tokens = self.tokenizer(
            styled_captions,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=77
        ).input_ids.to(self.device)

        text_embeddings = self.text_encoder(tokens)[0]
        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, int(1000 * self.train_strength), (B,), device=self.device, dtype=torch.long)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings)['sample']
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        if isinstance(style_labels[0], list):
            self.log('train_loss_fusion', loss, on_step=True, on_epoch=True, batch_size=B)
        else:
            style_name = style_labels[0] if isinstance(style_labels[0], str) else 'unknown'
            self.log(f'train_loss_{style_name}', loss, on_step=True, on_epoch=True, batch_size=B)

        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def configure_optimizers(self):
        trainable_params = []
        unet_param_count = 0
        for param in self.unet.parameters():
            if param.requires_grad:
                trainable_params.append(param)
                unet_param_count += param.numel()

        token_param_count = 0
        for token in self.style_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id < len(self.text_encoder.text_model.embeddings.token_embedding.weight):
                embedding_param = self.text_encoder.text_model.embeddings.token_embedding.weight[token_id]
                if embedding_param.requires_grad:
                    trainable_params.append(embedding_param)
                    token_param_count += embedding_param.numel()

        print(f"Trainable parameters - UNet LoRA: {unet_param_count:,}, Style tokens: {token_param_count:,}")

        unet_params = [p for p in self.unet.parameters() if p.requires_grad]
        token_params = []
        for token in self.style_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            embedding_param = self.text_encoder.text_model.embeddings.token_embedding.weight[token_id]
            if embedding_param.requires_grad:
                token_params.append(embedding_param)

        param_groups = []
        if unet_params:
            param_groups.append({'params': unet_params, 'lr': self.lr})
        if token_params:
            param_groups.append({'params': token_params, 'lr': self.lr * 5.0})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def save_style_tokens(self, save_path: str):
        style_embeddings = {}
        for token in self.style_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id < len(self.text_encoder.text_model.embeddings.token_embedding.weight):
                embedding = self.text_encoder.text_model.embeddings.token_embedding.weight[token_id]
                style_embeddings[token] = embedding.detach().cpu()

        torch.save(style_embeddings, save_path)
        print(f"Style token embeddings saved to {save_path}")

        if len(self.style_tokens) > 1:
            similarities = {}
            embeddings_list = list(style_embeddings.values())
            for i, token1 in enumerate(self.style_tokens):
                for j, token2 in enumerate(self.style_tokens[i + 1:], i + 1):
                    sim = torch.cosine_similarity(embeddings_list[i], embeddings_list[j], dim=0)
                    similarities[f"{token1}_{token2}"] = sim.item()

            print("Style token similarities:", similarities)
