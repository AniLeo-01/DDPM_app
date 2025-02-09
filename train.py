from pathlib import Path
from torch.optim import Adam
import torch
from tqdm import tqdm
from .src.model import UNet
from torchvision.utils import save_image
from .src.utils import p_losses, sample, num_to_groups
from .src.dataloader import dataloader
from .src.config import *


results_folder = Path("./results")
results_folder.mkdir(exist_ok=True)

model = UNet(
  dim=image_size,
  channels=channels,
  dim_mults=(1, 2, 4,),
)
model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
  print(f"Epoch {epoch+1} / {epochs}")
  for step, batch in enumerate(dataloader):
    batch_size_ = batch["pixel_values"].shape[0]
    batch_data = batch["pixel_values"].to(device)
    t = torch.randint(0, timesteps, (batch_size_,), device=device).long()

    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
      loss = p_losses(model, batch_data, t, loss_type="huber")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
      print(f"Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # save generated images
    if step != 0 and step % save_and_sample_every == 0:
      milestone = step // save_and_sample_every
      batches = num_to_groups(4, batch_size_)
      all_images_list = list(map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels), batches))
      all_images = torch.cat(all_images_list, dim=0)
      all_images = (all_images + 1) * 0.5
      save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
