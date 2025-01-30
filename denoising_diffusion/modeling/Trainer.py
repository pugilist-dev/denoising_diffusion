import pathlib

import torch
from tqdm import tqdm
from torchvision import utils
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch.optim import AdamW

from denoise_diffusion.utils import cycle
from denoising_diffusion.modeling.dataset import CelebADataset
from denoising_diffusion.modeling.models.diffusion_process import DiffusionModel

class Trainer:
    def __init__(
            self,
            diffusion_model: DiffusionModel,
            data_loc: str,
            results_loc: str,
            train_batch_size: int = 16,
            augment_horizontal_flip: bool = True,
            train_lr: float = 2e-4,


    ):
        pass