from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from .config import batch_size

dataset = load_dataset("fashion_mnist")

transform = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Lambda(lambda t: (t * 2) - 1)
])

def transforms(samples):
    samples['pixel_values'] = [transform(image.convert("L")) for image in samples['image']]
    del samples['image']
    return samples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

dataloader = DataLoader(transformed_dataset['train'], batch_size=batch_size, shuffle=True)


