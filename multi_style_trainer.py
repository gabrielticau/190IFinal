import argparse
import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset
from diffusers import StableDiffusionPipeline
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from data.monet import MonetDataset
from train.gated_lora_module import GatedLoRAModule


def str2bool(x):
    if x.lower() in ['true', '1', 'yes']:
        return True
    elif x.lower() in ['false', '0', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MultiStyleDataset:
    """
    Wrapper to combine multiple style datasets with style labels.
    """
    def __init__(self, dataset_configs):
        self.datasets = []
        self.style_labels = []
        
        for config in dataset_configs:
            dataset = MonetDataset(
                data_path=config['path'],
                center_crop=config.get('center_crop', False)
            )
            self.datasets.append(dataset)
            
        # Create combined dataset
        self.combined_dataset = ConcatDataset(self.datasets)
    
    def get_dataloader(self, batch_size, num_workers=8):
        return DataLoader(
            dataset=self.combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )


def main():
    parser = argparse.ArgumentParser(description='Train multi-style LoRA with gating')
    
    # Model parameters
    parser.add_argument('--pretrained_path', type=str, required=True,
                       help='Path to pretrained Stable Diffusion model')
    
    # Dataset parameters
    parser.add_argument('--style1_path', type=str, required=True,
                       help='Path to first style dataset')
    parser.add_argument('--style1_prefix', type=str, default='A Picasso painting, ',
                       help='Text prefix for first style')
    parser.add_argument('--style2_path', type=str, required=True,
                       help='Path to second style dataset')
    parser.add_argument('--style2_prefix', type=str, default='A Van Gogh painting, ',
                       help='Text prefix for second style')
    parser.add_argument('--center_crop', type=str2bool, default=False,
                       help='Use center crop instead of random crop')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # Training parameters
    parser.add_argument('--train_strength', type=float, required=True,
                       help='Training noise strength')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate for LoRA adapters')
    parser.add_argument('--gating_lr', type=float, default=1e-4,
                       help='Learning rate for style gating module')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--grad_accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum training epochs')
    
    # Loss weights
    parser.add_argument('--style_consistency_weight', type=float, default=0.1,
                       help='Weight for style consistency loss')
    parser.add_argument('--diversity_weight', type=float, default=0.05,
                       help='Weight for style diversity loss')
    
    # Output
    parser.add_argument('--save_name', type=str, required=True,
                       help='Name for saving model checkpoints')
    parser.add_argument('--project_name', type=str, default='multi_style_lora',
                       help='WandB project name')
    
    args = parser.parse_args()
    
    # Load pretrained pipeline
    print("Loading pretrained Stable Diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_path)
    
    # Define LoRA configurations for each style
    lora_configs = [
        {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'init_lora_weights': 'gaussian',
            'target_modules': ['to_k', 'to_q', 'to_v', 'to_out.0'],
            'lora_dropout': args.lora_dropout,
            'bias': 'none'
        },
        {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'init_lora_weights': 'gaussian',
            'target_modules': ['to_k', 'to_q', 'to_v', 'to_out.0'],
            'lora_dropout': args.lora_dropout,
            'bias': 'none'
        }
    ]
    
    # Create multi-style model
    print("Creating multi-style LoRA module...")
    model = GatedLoRAModule(
        pipeline=pipeline,
        style_prefixes=[args.style1_prefix, args.style2_prefix],
        lora_configs=lora_configs,
        lr=args.learning_rate,
        train_strength=args.train_strength,
        gating_lr=args.gating_lr,
        style_consistency_weight=args.style_consistency_weight,
        diversity_weight=args.diversity_weight
    )
    
    # Create datasets
    print("Loading datasets...")
    dataset_configs = [
        {'path': args.style1_path, 'center_crop': args.center_crop},
        {'path': args.style2_path, 'center_crop': args.center_crop}
    ]
    
    multi_dataset = MultiStyleDataset(dataset_configs)
    dataloader = multi_dataset.get_dataloader(batch_size=args.batch_size)
    
    # Logger and callbacks
    logger = WandbLogger(
        project=args.project_name,
        name=args.save_name
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./multi_style_ckpt/{args.save_name}',
        filename='{epoch}-{train_loss:.4f}',
        monitor='train_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    # Trainer
    print("Starting training...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision='bf16-mixed',
        accumulate_grad_batches=args.grad_accumulation,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    
    # Train
    trainer.fit(model, train_dataloaders=dataloader)
    
    # Save final model
    print("Saving final model...")
    model.save_style_adapters(f'./multi_style_final/{args.save_name}')
    
    print("Training completed!")


if __name__ == "__main__":
    main()