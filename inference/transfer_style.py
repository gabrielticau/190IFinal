import argparse
from PIL import Image
from peft import PeftModel
from torchvision import transforms
from diffusers import AutoPipelineForImage2Image
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--ip_adapter_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--image_cond_scale', type=float, required=True)
    parser.add_argument('--strength', type=float, required=True)
    parser.add_argument('--infer_steps', type=int, required=True)
    parser.add_argument('--save_path', type=str, default='styled_image.jpg')
    args = vars(parser.parse_args())

    # Load pretrained pipeline
    pipeline = AutoPipelineForImage2Image.from_pretrained(args['pretrained_path'])
    pipeline.to('cuda')

    # Load IP-Adapter
    pipeline.load_ip_adapter(
        args['ip_adapter_path'],
        subfolder='models',
        weight_name='ip-adapter_sd15.bin'
    )
    pipeline.set_ip_adapter_scale(args['image_cond_scale'])

    # Load LoRA model
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args['lora_path'])

    # Load and process image
    im = Image.open(args['image_path'])

    image_transforms_ip = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    im_init = image_transforms(im)
    im_ip = image_transforms_ip(im).unsqueeze(0)

    # Ensure style tokens are available
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    device = pipeline.device

    tokenizer.add_tokens(["<monet>", "<ukiyo_e>"])
    text_encoder.resize_token_embeddings(len(tokenizer))

    monet_id = tokenizer.convert_tokens_to_ids("<monet>")
    ukiyo_id = tokenizer.convert_tokens_to_ids("<ukiyo_e>")

    # Blend style tokens: 70% ukiyo_e, 30% monet
    with torch.no_grad():
        embedding_layer = text_encoder.text_model.embeddings.token_embedding
        monet_embedding = embedding_layer.weight[monet_id]
        ukiyo_embedding = embedding_layer.weight[ukiyo_id]

        fused_embedding = 0 * monet_embedding + 1 * ukiyo_embedding

        # Overwrite <monet> embedding with blended one
        embedding_layer.weight[monet_id] = fused_embedding

    # Inference
    style_image = pipeline(
        prompt=args['prompt'],
        image=im_init,
        ip_adapter_image=im_ip,
        num_inference_steps=args['infer_steps'],
        strength=args['strength'],
    ).images[0]

    # Save result
    style_image.save(args['save_path'])

if __name__ == "__main__":
    main()
