from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import timesteps
from tqdm import tqdm

def exists(x):
  return x is not None

def default(val, d):
  return val if exists(val) else d() if isfunction(d) else d

def num_to_groups(num, divisor):
  groups = num // divisor
  remainder = num % divisor
  arr = [divisor] * groups
  if remainder > 0:
      arr.append(remainder)
  return arr

def extract(a, t, x_shape):
  batch_size = t.shape[0]
  out = a.gather(-1, t.cpu())
  return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
  if noise is None:
    noise = torch.randn_like(x_start)
  sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
  sqrt_one_minus_alphas_cumprod_t = extract(
    sqrt_one_minus_alphas_cumprod, t, x_start.shape
  )
  return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def cosine_beta_schedule(timesteps, s=0.008):
  """
  cosine schedule as proposed in https://arxiv.org/abs/2102.09672
  """
  steps = timesteps + 1
  x = torch.linspace(0, timesteps, steps)
  alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
  beta_start = 0.0001
  beta_end = 0.02
  return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
  beta_start = 0.0001
  beta_end = 0.02
  return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
  beta_start = 0.0001
  beta_end = 0.02
  betas = torch.linspace(-6, 6, timesteps)
  return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

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
  
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
  if noise is None:
    noise = torch.randn_like(x_start)

  x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
  predicted_noise = denoise_model(x_noisy, t)

  if loss_type == 'l1':
    loss = F.l1_loss(noise, predicted_noise)
  elif loss_type == 'l2':
    loss = F.mse_loss(noise, predicted_noise)
  elif loss_type == "huber":
    loss = F.smooth_l1_loss(noise, predicted_noise)
  else:
    raise NotImplementedError()
  return loss

@torch.no_grad()
def p_sample(model, x, t, t_index):
  betas_t = extract(betas, t, x.shape)
  sqrt_one_minus_alphas_cumprod_t = extract(
    sqrt_one_minus_alphas_cumprod, t, x.shape
  )
  sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

  # Equation 11 in the paper
  # Use our model (noise predictor) to predict the mean
  model_mean = sqrt_recip_alphas_t * (
    x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
  )

  if t_index == 0:
    return model_mean
  else:
    posterior_variance_t = extract(posterior_variance, t, x.shape)
    noise = torch.randn_like(x)
    # Algorithm 2 line 4:
    return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps=timesteps):
  device = next(model.parameters()).device
  b = shape[0]
  img = torch.randn(shape, device=device)

  for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
    img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)

  return img
    
@torch.no_grad()
def p_sample_loop_chain(model, shape, timesteps=1000):
  """
  Returns a 5D tensor of shape:
  (timesteps, batch_size, channels, height, width)
  capturing each intermediate step of diffusion.
  """
  device = next(model.parameters()).device
  b = shape[0]
  img = torch.randn(shape, device=device)
  imgs = []  # to store each step

  # We go from (timesteps-1) down to 0
  for i in reversed(range(timesteps)):
    t = torch.full((b,), i, device=device, dtype=torch.long)
    img = p_sample(model, img, t, i)
    imgs.append(img.clone())  # clone to avoid in-place ops

  # Right now, imgs[0] is the final step, imgs[-1] is the first step
  # Reverse so that imgs[0] is the 1st step, and imgs[-1] is the final step
  imgs.reverse()

  return torch.stack(imgs, dim=0)  # shape => (timesteps, batch_size, channels, H, W)

@torch.no_grad()
def sample_chain(model, image_size, batch_size=16, channels=3, timesteps=1000):
  shape = (batch_size, channels, image_size, image_size)
  return p_sample_loop_chain(model, shape, timesteps=timesteps)

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
  return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))