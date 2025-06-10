import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MonetDataset(Dataset):
    """Lazy load Monet dataset based on the provided data_path"""
    def __init__(self, data_path: str, center_crop: bool = False):
        self.image_base_path = os.path.join(data_path, 'photo')
        self.caption_base_path = os.path.join(data_path, 'caption')

        self.image_names = os.listdir(self.image_base_path)
        self.captions = os.listdir(self.caption_base_path)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256) if center_crop else transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx) -> np.array:
        # image
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_base_path, image_name)

        im = Image.open(image_path)
        # Convert to RGB to ensure 3 channels (handles grayscale, RGBA, etc.)
        im = im.convert('RGB')
        im = self.image_transforms(im)

        # caption
        caption_file = self.captions[idx]
        caption_path = os.path.join(self.caption_base_path, caption_file)

        with open(caption_path, 'r') as f:
            cap = f.read().strip()

        return im, cap