from pathlib import Path
from torch.optim import Adam
import torch
from tqdm import tqdm
from .modules import UNet
from torchvision.utils import save_image
from .utils import p_losses, sample
from .dataloader import dataloader
from .config import *

def num_to_groups(num, divisor):
  groups = num // divisor
  remainder = num % divisor
  arr = [divisor] * groups
  if remainder > 0:
    arr.append(remainder)
  return arr

results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)

device = device

model = UNet(
  dim=image_size,
  channels=channels,
  dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
epochs = 6

for epoch in tqdm(range(epochs), desc='Training epochs'):
  for step, batch in tqdm(enumerate(dataloader), desc=f'Epoch {epoch+1}', total=len(dataloader)):
    optimizer.zero_grad()

    batch_size = batch["pixel_values"].shape[0]
    batch = batch["pixel_values"].to(device)

    # Algorithm 1 line 3: sample t uniformally for every example in the batch
    t = torch.randint(0, timesteps, (batch_size,), device=device).long()

    loss = p_losses(model, batch, t, loss_type="huber")

    if step % 100 == 0:
      print("Loss:", loss.item())

    loss.backward()
    optimizer.step()

    # save generated images
    if step != 0 and step % save_and_sample_every == 0:
      milestone = step // save_and_sample_every
      batches = num_to_groups(4, batch_size)
      all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
      all_images = torch.cat(all_images_list, dim=0)
      all_images = (all_images + 1) * 0.5
      save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
