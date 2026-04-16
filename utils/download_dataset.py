import torch
import torchvision
from torchvision import datasets, models, transforms

class dataset():
  @staticmethod
  def getDataset(bt):
    image_transforms = transforms.Compose(
      [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
      ]
    )

    return datasets.CIFAR10(
      "../data",
      train=bt,
      download=True,  
      transform=image_transforms
    )