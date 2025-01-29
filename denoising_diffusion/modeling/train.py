from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Deep Learning Libraries
import torch

from denoising_diffusion.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    train_config
                                        )
from denoising_diffusion.modeling.models.unet import UNet

if __name__ == "__main__":
    if torch.has_mps:
        torch.mps.empty_cache()
        print("MPS cache cleared.")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    else:
        print("Accelerators not available.")
    model_name = train_config["model_name"]
    unet_config = train_config["model_config"]
    trainer_config = train_config["trainer_config"]
    diffusion_config = train_config["diffusion_config"]

    unet = getattr(UNet, model_name)
