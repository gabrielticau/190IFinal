import os
import argparse


class CaptionProcessor:
    """Extract Monet picture name as caption and write captions to output_path"""
    def __init__(self, data_path: str, out_path: str):
        self.data_path = data_path
        self.out_path = out_path
        self.image_names = os.listdir(data_path)
    
    def generate_caption(self):
        for image_name in self.image_names:
            caption_path = os.path.join(self.out_path, image_name)
            caption = image_name.split('_')[0].replace('(1)', '')[:-4]
            caption = caption.replace('-', ' ')

            # write file
            f = open(f'{caption_path}.txt', 'w')
            f.write(caption)
            f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--monet_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = vars(parser.parse_args())

    cp = CaptionProcessor(args['monet_path'], args['output_path'])
    cp.generate_caption()