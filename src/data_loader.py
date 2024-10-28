# src/data_loader.py
import torch
from torchvision import datasets, transforms
from config import Config

def load_data():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_data = datasets.CIFAR10(root=Config.DATA_PATH, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root=Config.DATA_PATH, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    return train_loader, test_loader
