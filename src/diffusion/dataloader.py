from datasets import load_dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from src.config import batch_size, image_size, channels
from io import BytesIO
from PIL import Image

dataset = load_dataset("fashion_mnist")

transform = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Lambda(lambda t: (t * 2) - 1)
])

def transform_samples(samples):
  samples['pixel_values'] = [transform(image.convert("L")) for image in samples['image']]
  del samples['image']
  return samples

transformed_dataset = dataset.with_transform(transform_samples).remove_columns("label")

dataloader = DataLoader(
  transformed_dataset['train'],
  batch_size=batch_size,
  shuffle=True,
  pin_memory=True,       # speed up host-to-device transfers
  num_workers=4          # adjust based on your system
)

# New custom dataloader
class CustomImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.images = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(BytesIO(self.images[idx].read())).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image}

def get_custom_dataloader(uploaded_files, batch_size, device):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=channels),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = CustomImageDataset(uploaded_files, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)