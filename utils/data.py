import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

def load_data(data_name, data_path, batch_size):
    if data_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=f'{data_path}/dataset', train=True, download=False, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=f'{data_path}/dataset', train=False, download=False, transform=transform)

    elif data_name == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BILINEAR),
            transforms.CenterCrop(size=(224, 224)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD_DEV),
        ])
        train_dataset = torchvision.datasets.ImageNet(root=f'{data_path}/dataset/imagenet2012', split='train', transform=transform)
        test_dataset = torchvision.datasets.ImageNet(root=f'{data_path}/dataset/imagenet2012', split='val', transform=transform)

    else:
        raise ValueError("Unsupported dataset. Please choose either 'CIFAR10' or 'ImageNet'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


