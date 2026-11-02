import torch
import torch.nn as nn
from torchvision import models


def build_model(in_channels=11):
    """ImageNet ResNet-18 adapted to 11 Sentinel-2 bands, with one output (debris / no debris)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old = model.conv1.weight.data  # (64, 3, 7, 7)
    conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        # Sentinel-2 bands 2, 3, 4 are blue, green, red: reuse the matching RGB filters,
        # and start every other band from the average RGB filter
        conv.weight[:] = old.mean(dim=1, keepdim=True)
        conv.weight[:, 1] = old[:, 2]
        conv.weight[:, 2] = old[:, 1]
        conv.weight[:, 3] = old[:, 0]
    model.conv1 = conv
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
