import argparse
import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline
from lightning.pytorch.loggers import WandbLogger

from data.multi_style_dataset import MultiStyleDataset
from train.lit_module import DiffusionLoRAModule

def str2bool(x):
    if x == "true":
        return True
    elif x == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('true or false expected.')

def custom_collate_fn(batch):
    """Custom collate function to handle mixed style data types"""
    images = []
    captions = []
    styles = []
    
    for item in batch:
        if len(item) == 3:
            image, caption, style = item
        else:
            image, caption = item
            style = 'monet'  # default style
            
        images.append(image)
        captions.append(caption)
        styles.append(style)
    
    # Stack images into a tensor
    images = torch.stack(images)
    
    # Keep captions and styles as lists (they'll be processed in the model)
    return images, captions, styles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--monet_path', type=str, required=True, help='Path to Monet images.')
    parser.add_argument('--ukiyo_e_path', type=str, required=True, help='Path to Ukiyo-e images.')
    parser.add_argument('--center_crop', type=str2bool, default=False)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--train_strength', type=float, required=True)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accumulation', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--multi_style_prob', type=float, default=0.3)
    parser.add_argument('--save_name', type=str, required=True)

    args = vars(parser.parse_args())

    # Use both paths directly
    style_paths = {
        'monet': args['monet_path'],
        'ukiyo_e': args['ukiyo_e_path']
    }
    print(f"Using provided paths for styles: {list(style_paths.keys())}")

    style_tokens = [f'<{style}>' for style in style_paths.keys()]
    print(f"Style tokens: {style_tokens}")

    pipeline = StableDiffusionPipeline.from_pretrained(args['pretrained_path'])

    lora_config = LoraConfig(
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        init_lora_weights='gaussian',
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        lora_dropout=args['lora_dropout'],
        bias='none'
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    model = DiffusionLoRAModule(
        pipeline=pipeline, 
        lr=args['learning_rate'], 
        train_strength=args['train_strength'],
        style_tokens=style_tokens,
        multi_style_prob=args['multi_style_prob']
    )

    dataset = MultiStyleDataset(
        data_paths=style_paths,
        center_crop=args['center_crop'],
        multi_style_prob=args['multi_style_prob']
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=0,  # Reduced from 4 to avoid worker issues during debugging
        persistent_workers=False,  # Disabled for debugging
        collate_fn=custom_collate_fn  # Use our custom collate function
    )

    logger = WandbLogger(
        project='multi-style-diffusion',
        name=f"{args['save_name']}_{'_'.join(style_paths.keys())}"
    )

    trainer = pl.Trainer(
        max_epochs=args['max_epochs'],
        precision='bf16-mixed',
        accumulate_grad_batches=args['grad_accumulation'],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=None,
        enable_checkpointing=True,
        default_root_dir=f'./checkpoints/{args["save_name"]}'
    )

    trainer.fit(model, train_dataloaders=dataloader)

    save_name = args['save_name']
    os.makedirs(f'./lora_ckpt/{save_name}', exist_ok=True)
    model.unet.save_pretrained(f'./lora_ckpt/{save_name}')
    model.save_style_tokens(f'./lora_ckpt/{save_name}/style_tokens.pt')
    model.tokenizer.save_pretrained(f'./lora_ckpt/{save_name}/tokenizer')

    print(f"Training complete! Saved to ./lora_ckpt/{save_name}/")
    print(f"Style tokens: {style_tokens}")

if __name__ == "__main__":
    main()