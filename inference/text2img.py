import argparse
from peft import PeftModel
from diffusers import StableDiffusionPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--infer_steps', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='output.jpg')
    args = vars(parser.parse_args())

    # Load pretrained model
    pipeline = StableDiffusionPipeline.from_pretrained(args['pretrained_path'])
    pipeline.to('cuda')

    # Load LoRA
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args['lora_path'])

    # Inference
    style_image = pipeline(prompt=args['prompt'], num_inference_steps=args['infer_steps']).images[0]

    # Save generated image
    style_image.save(args['save_path'])

if __name__ == "__main__":
    main()

