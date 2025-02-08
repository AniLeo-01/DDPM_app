from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from PIL import Image
import requests
import numpy as np
from src.utils import q_sample
import torch
from src.config import image_size, channels, timesteps, device

import matplotlib.pyplot as plt

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
transform = Compose([
  Resize(image_size),
  CenterCrop(image_size),
  ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
  Lambda(lambda t: (t * 2) - 1),
])

reverse_transform = Compose([
  Lambda(lambda t: (t + 1) / 2),
  Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
  Lambda(lambda t: t * 255.),
  Lambda(lambda t: t.numpy().astype(np.uint8)),
  ToPILImage(),
])

# print(x_start.shape)

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(12, 8), nrows=num_rows, ncols=num_cols, squeeze=False)  # Changed from (200,200)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

if __name__ == "__main__":
  # take time step
  x_start = transform(image).unsqueeze(0)
  plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])
  plt.show()