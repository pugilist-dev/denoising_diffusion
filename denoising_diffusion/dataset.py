from pathlib import Path
import shutil

from loguru import logger
from tqdm import tqdm

from denoising_diffusion.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

#KaggleHub to download data from Kaggle
import kagglehub

if __name__ == "__main__":
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset",
                                       force_download=True)
    print("Downloaded at: ", path)
    print("Moving data to Raw data directory: ", RAW_DATA_DIR)
    raw_data_dir = Path(RAW_DATA_DIR)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    for item in tqdm(Path(path).iterdir(), desc="Moving files"):
        dest = raw_data_dir / item.name
        if item.is_dir():
            shutil.move(str(item), str(dest))
        else:
            shutil.move(str(item), str(dest))

    print(f"All files moved to {RAW_DATA_DIR}")
