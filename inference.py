from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from .src.model import UNet
from .src.utils import extract, linear_beta_schedule
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from .src.config import image_size, channels, timesteps, device
from .src.utils import sample, sample_chain

model = UNet(
  dim=image_size,
  channels=channels,
  dim_mults=(1, 2, 4,),
)
model.to(device)

# sample 64 images
def sample_image(filename: str = "sample_image.png"):
  samples = sample(model, image_size=image_size, batch_size=64, channels=channels)
  samples = (samples + 1) * 0.5  # scale from [-1,1] to [0,1]

  # pick a random index from 0..63
  random_index = np.random.randint(0, 64)

  # If channels=1: shape = [batch_size, 1, H, W]
  # We'll select the random_index sample, the single channel, then reshape to HxW
  sample_image = samples[random_index, 0].detach().cpu().numpy()

  plt.imsave(filename, sample_image, cmap="gray")

def create_sample_grid_video(
  timesteps: int,
  grid_size=(10, 10),
  filename='diffusion.mp4',
  fps=10
):
  """
  Create an MP4 video from diffusion samples, arranged in a grid at each timestep.

  Args:
      samples (Tensor): shape (timesteps, batch_size, channels, H, W)
      timesteps (int): total number of diffusion steps
      channels (int): number of channels (1 for gray, 3 for RGB, etc.)
      image_size (int): height/width of each image
      grid_size (tuple): (rows, cols) for the grid, e.g. (10,10)
      filename (str): output video filename, e.g. 'diffusion.mp4'
      fps (int): frames per second
  """
  samples = sample_chain(
    model,
    image_size=image_size,
    batch_size=64,
    channels=channels,
    timesteps=timesteps
  )

  fig = plt.figure()
  frames = []

  # We'll assume we want grid_size[0] * grid_size[1] images = 100 for (10,10).
  num_images_needed = grid_size[0] * grid_size[1]
  # The batch dimension is samples.shape[1]
  batch_size = samples.shape[1]

  # If the batch is smaller than num_images_needed, only use batch_size images.
  num_images = min(num_images_needed, batch_size)

  # Slice out only the first num_images from each timestep
  # shape => (timesteps, num_images, channels, H, W)
  truncated_samples = samples[:, :num_images]

  for t in range(timesteps):
    # truncated_samples[t] => shape (num_images, channels, H, W)
    frame = truncated_samples[t]

    # If your samples are in [-1,1], scale to [0,1]
    frame = (frame + 1) * 0.5
    frame = frame.clamp(0, 1)

    # Use torchvision make_grid to create a (channels, gridH, gridW) tensor
    # nrow=grid_size[1] => number of columns in the final grid.
    grid = make_grid(frame, nrow=grid_size[1])
    # shape => (channels, gridH, gridW)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # Display in matplotlib
    plt.axis("off")
    imgplot = plt.imshow(grid_np, animated=True)
    frames.append([imgplot])

  # Build animation
  ani = animation.ArtistAnimation(fig, frames, interval=1000/fps, blit=True)

  # Save as MP4 using ffmpeg
  # Make sure you have ffmpeg installed or conda install -c conda-forge ffmpeg
  ani.save(filename, writer="ffmpeg", fps=fps)

  plt.close(fig)

if __name__ == "__main__":
  sample_image()
  create_sample_grid_video(timesteps)