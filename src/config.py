import torch

image_size=28
channels=1
timesteps=1000
batch_size=64
epochs=6
save_and_sample_every=500
device="cuda" if torch.cuda.is_available() else "cpu"
learning_rate=1e-3