from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Deep Learning Libraries
import torch

from denoising_diffusion.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    train_config
                                        )
from denoising_diffusion.modeling.models import (
    unet, diffusion_process
)

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

    unet = getattr(unet, model_name)
    model = unet(dim = unet_config["input"],
                 channels = unet_config["channels"],
                 dim_mults = tuple(unet_config["dim_mults"]),)
    diffusion_model = diffusion_process.DiffusionModel(
        model=model,
        image_size=diffusion_config["image_size"],
        beta_scheduler=diffusion_config["beta_scheduler"],
        timesteps=diffusion_config["timesteps"],
    )