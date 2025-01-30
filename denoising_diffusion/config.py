from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Training config
train_config = {
    "model_name": "UNet",
    "trainer_config":{
        "batch_size": 16,
        "train_lr":1e-4,
        "train_num_steps": 100000,
        "train_save_interval": 1000,
        "num_samples": 4,
    },
    "model_config": {
        "model_mapping": "UNet",
        "input": 64,
        "batch_size": 16,
        "dim_mults": [1, 2, 4, 8],
        "channels": 3,
    },
    "diffusion_config":{
        "timesteps": 1000,
        "beta_scheduler": "linear",
        "image_size": 128,
    }
}
