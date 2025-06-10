import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiStyleDataset(Dataset):
    """Dataset that handles multiple art styles and can mix them during training"""

    def __init__(self, data_paths: dict, center_crop: bool = False, multi_style_prob: float = 0.3):
        """
        Args:
            data_paths: Dict mapping style names to their data paths
            center_crop: Whether to use center crop instead of random crop
            multi_style_prob: Probability of mixing styles during training
        """
        self.style_names = list(data_paths.keys())
        self.multi_style_prob = multi_style_prob

        self.images = []
        self.captions = []
        self.styles = []

        print(f"Loading datasets for styles: {self.style_names}")

        for style, path in data_paths.items():
            image_path = path
            caption_path = os.path.join(path, 'caption')

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path not found: {image_path}")
            if not os.path.exists(caption_path):
                raise FileNotFoundError(f"Caption path not found: {caption_path}")

            image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} images for {style}")

            for img_name in image_files:
                img_fp = os.path.join(image_path, img_name)
                cap_fp = os.path.join(caption_path, img_name + '.txt')

                if not os.path.exists(cap_fp):
                    cap_fp = os.path.join(caption_path, img_name.rsplit('.', 1)[0] + '.txt')

                if os.path.exists(cap_fp):
                    with open(cap_fp, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                else:
                    caption = self._generate_fallback_caption(img_name, style)

                self.images.append(img_fp)
                self.captions.append(caption)
                self.styles.append(style)

        print(f"Total dataset size: {len(self.images)} images")
        print(f"Style distribution: {dict(zip(*np.unique(self.styles, return_counts=True)))}")

        self.image_transforms = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512) if center_crop else transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _generate_fallback_caption(self, img_name: str, style: str) -> str:
        base_name = img_name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ')
        if style == 'monet':
            return f"impressionist painting of {base_name}"
        elif style == 'ukiyo_e':
            return f"japanese woodblock print of {base_name}"
        else:
            return f"artwork depicting {base_name}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.image_transforms(image)
        except Exception as e:
            print(f"Error loading image {self.images[idx]}: {e}")
            return self.__getitem__(random.randint(0, len(self.images) - 1))

        caption = self.captions[idx]
        style = self.styles[idx]

        if random.random() < self.multi_style_prob and len(self.style_names) > 1:
            other_styles = [s for s in self.style_names if s != style]
            if other_styles:
                style = [style, random.choice(other_styles)]

        return image, caption, style


class MonetUkiyoeDataset(MultiStyleDataset):
    def __init__(self, monet_path: str, ukiyo_e_path: str, center_crop: bool = False, multi_style_prob: float = 0.4):
        data_paths = {
            'monet': monet_path,
            'ukiyo_e': ukiyo_e_path
        }
        super().__init__(data_paths, center_crop, multi_style_prob)

        self.fusion_captions = [
            "serene landscape with flowing water",
            "garden scene with delicate flowers",
            "peaceful nature scene",
            "traditional architecture in nature",
            "water lilies and reflections",
            "mountain landscape at different times of day",
            "seasonal nature scene",
            "bridges over tranquil waters"
        ]

    def __getitem__(self, idx):
        image, caption, style = super().__getitem__(idx)
        if isinstance(style, list) and random.random() < 0.3:
            caption = random.choice(self.fusion_captions)
        return image, caption, style
