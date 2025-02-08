import torch
from src.modules import UNet
from src.utils import extract, linear_beta_schedule
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from src.config import image_size, channels, timesteps, device
from src.utils import sample

timesteps = 1000
betas = linear_beta_schedule(timesteps)

#definining the alpha schedule
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value = 1.0)
sqrt_recip_alphas = torch.sqrt(1.0/alphas)

# q(x_t | x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

model = UNet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)

if __name__ == "__main__":
  samples = sample(model, image_size=image_size, batch_size=64, channels=channels)
  save_image(samples, "samples.png", nrow=6)