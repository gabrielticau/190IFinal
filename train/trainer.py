import argparse
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline
from lightning.pytorch.loggers import WandbLogger

from data.monet import MonetDataset
from train.lit_module import DiffusionLoRAModule

def str2bool(x):
    if x == "true":
        return True
    elif x == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('true or false expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--monet_path', type=str, required=True)
    parser.add_argument('--center_crop', type=str2bool, default=False)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--train_strength',  type=float, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_name', type=str, required=True)
    args = vars(parser.parse_args())

    # Pre-trained model
    pipeline = StableDiffusionPipeline.from_pretrained(args['pretrained_path'])

    # PEFT LoRA injection
    lora_config = LoraConfig(
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        init_lora_weights='gaussian',
        target_modules=['to_k', 'to_q', 'to_v', 'to_out.0', 'add_k_proj', 'add_v_proj'],
        lora_dropout=args['lora_dropout'],
        bias='none'
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    # Trainer
    model = DiffusionLoRAModule(
        pipeline=pipeline, 
        lr=args['learning_rate'], 
        train_strength=args['train_strength']
    )

    # Data
    monet_ds = MonetDataset(
        data_path=args['monet_path'], 
        center_crop=args['center_crop']
    )
    monet_dl = DataLoader(
        dataset=monet_ds,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=8
    )

    # Logger
    logger = WandbLogger(
        project='monet'
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=args['max_epochs'],
        precision='bf16',
        accumulate_grad_batches=args['grad_accumulation'],
        logger=logger
    )
    trainer.fit(model, train_dataloaders=monet_dl)

    # Save LoRA weights
    save_name = args['save_name']
    model.unet.save_pretrained(f'./lora_ckpt/{save_name}')


if __name__ == "__main__":
    main()