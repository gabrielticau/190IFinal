import argparse
from PIL import Image
from peft import PeftModel
from torchvision import transforms
from diffusers import AutoPipelineForImage2Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--lora_path_1', type=str, required=True, help='Path to first LoRA adapter')
    parser.add_argument('--lora_path_2', type=str, required=True, help='Path to second LoRA adapter')
    parser.add_argument('--lora_weight_1', type=float, default=0.5, help='Weight for first LoRA (default: 0.5)')
    parser.add_argument('--lora_weight_2', type=float, default=0.5, help='Weight for second LoRA (default: 0.5)')
    parser.add_argument('--combination_type', type=str, default='linear', 
                        choices=['linear', 'dare_linear', 'ties', 'magnitude_prune'],
                        help='Method for combining LoRAs (default: linear)')
    parser.add_argument('--ip_adapter_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--image_cond_scale', type=float, required=True)
    parser.add_argument('--strength', type=float, required=True)
    parser.add_argument('--infer_steps', type=int, required=True)
    parser.add_argument('--save_path', type=str, default='styled_image.jpg')
    args = vars(parser.parse_args())

    # Load pretrained model
    pipeline = AutoPipelineForImage2Image.from_pretrained(args['pretrained_path'])
    pipeline.to('cuda')

    # Load IP-Adapter
    pipeline.load_ip_adapter(
        args['ip_adapter_path'], 
        subfolder='models', 
        weight_name='ip-adapter_sd15.bin'
    )
    pipeline.set_ip_adapter_scale(args['image_cond_scale'])

    # Load first LoRA and create PeftModel
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args['lora_path_1'], adapter_name="lora_1")
    
    # Load second LoRA as additional adapter
    pipeline.unet.load_adapter(args['lora_path_2'], adapter_name="lora_2")
    
    # Merge adapters using add_weighted_adapter method
    pipeline.unet.add_weighted_adapter(
        adapters=["lora_1", "lora_2"],
        weights=[args['lora_weight_1'], args['lora_weight_2']],
        combination_type=args['combination_type'],  # Can also use "dare_linear", "ties", etc.
        adapter_name="merged_lora"
    )
    
    # Set the merged adapter as active
    pipeline.unet.set_adapters("merged_lora")

    # Load content image
    im = Image.open(args['image_path'])
    
    # Image Transform
    image_transforms_ip = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )
    image_transforms = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    im_init = image_transforms(im)
    im_ip = image_transforms_ip(im).unsqueeze(0)

    # Inference
    style_image = pipeline(
        prompt=args['prompt'], 
        image=im_init,
        ip_adapter_image=im_ip,
        num_inference_steps=args['infer_steps'],
        strength=args['strength'],
        cross_attention_kwargs={"scale": 1.0}  # LoRA scale is controlled by adapter weights
    ).images[0]

    # Save generated image
    style_image.save(args['save_path'])
    print(f"Generated image saved to: {args['save_path']}")
    print(f"LoRA combination: {args['lora_weight_1']:.2f} (LoRA 1) + {args['lora_weight_2']:.2f} (LoRA 2) using {args['combination_type']} method")

if __name__ == "__main__":
    main()