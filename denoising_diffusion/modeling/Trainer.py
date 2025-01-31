import pathlib
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision import utils
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torch.optim import AdamW

from denoising_diffusion.utils import cycle
from denoising_diffusion.modeling.dataset import CelebADataset
from denoising_diffusion.modeling.models.diffusion_process import DiffusionModel

class Trainer:
    def __init__(
            self,
            diffusion_model: DiffusionModel,
            data_loc: Path,
            results_loc: Path,
            train_batch_size: int = 16,
            augment_horizontal_flip: bool = True,
            train_lr: float = 2e-4,
            train_num_steps: int = 100000,
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            adam_betas: tuple[float, float] = (0.9, 0.999),
            save_and_sample_every: int = 1000,
            num_samples: int = 4,
            save_best_and_latest: bool = False,


    ):
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.step = 0
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.ds = CelebADataset(data_loc,
                                 image_size=self.image_size,
                                   augment_horizontal_flip=augment_horizontal_flip)
        self.dl = cycle(DataLoader(self.ds,
                                    batch_size=self.batch_size,
                                      shuffle=True, num_workers=0, pin_memory=True,
                                      ))
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        self.ema = EMA(diffusion_model,
                        beta=ema_decay,
                          update_every=ema_update_every)
        self.ema.to(self.device)
        self.results_loc = pathlib.Path(results_loc)
        self.results_loc.mkdir(parents=True, exist_ok=True)
        self.save_best_and_latest = save_best_and_latest
    
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def save(self, name):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
        }

        torch.save(data, str(self.results_loc / f"model-{name}.pt"))

    def load(self, name):
        data = torch.load(
            str(self.results_loc / f"model-{name}.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(data["model"])
        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                data = next(self.dl).to(self.device)
                loss = self.model(data)
                total_loss += loss.item()

                loss.backward()

                pbar.set_description(f"loss: {total_loss:.4f}")

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    self.ema.ema_model.eval()

                    with torch.inference_mode():
                        milestone = self.step // self.save_and_sample_every
                        sampled_imgs = self.ema.ema_model.sample(
                            batch_size=self.num_samples
                        )
                    
                    for ix, sampled_imgs in enumerate(sampled_imgs):
                        utils.save_image(
                            sampled_imgs,
                            str(self.result_loc / f"sample_{milestone}_{ix}.png"),
                        )
                    
                    self.save(milestone)
                    torch.cuda.empty_cache()
                pbar.update(1)
