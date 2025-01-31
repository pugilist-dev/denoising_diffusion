from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from glob import glob
import torch
import torch.nn as nn

class CelebADataset(Dataset):
    def __init__(
            self,
            folder: Path,
            image_size: int = 64,
            augment_horizontal_flip: bool = False,
    ):
        self.paths = list(glob(str(folder / "*.jpg")))
        self.img_size = image_size

        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img