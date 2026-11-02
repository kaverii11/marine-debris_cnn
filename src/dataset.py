"""MARIDA patch-level dataset: does a 256x256 Sentinel-2 patch contain marine debris?"""
import json
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Index of "Marine Debris" in labels_mapping.txt (class 1 of 15 in the MARIDA paper)
MARINE_DEBRIS = 0


def patch_path(root, name):
    """'S2_1-12-19_48MYU_0' -> <root>/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif"""
    scene = name.rsplit("_", 1)[0]
    return os.path.join(root, "patches", scene, name + ".tif")


def read_split(root, split):
    with open(os.path.join(root, "splits", f"{split}_X.txt")) as f:
        return ["S2_" + line.strip() for line in f if line.strip()]


def read_labels(root):
    with open(os.path.join(root, "labels_mapping.txt")) as f:
        mapping = json.load(f)
    return {k.replace(".tif", ""): v for k, v in mapping.items()}


def load_patch(path):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)  # (11, 256, 256) reflectance bands
    return np.nan_to_num(img)


class MaridaPatches(Dataset):
    def __init__(self, root, split, mean=None, std=None, augment=False):
        labels = read_labels(root)
        self.items = [(patch_path(root, n), float(labels[n][MARINE_DEBRIS])) for n in read_split(root, split)]
        self.mean, self.std, self.augment = mean, std, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = load_patch(path)
        if self.mean is not None:
            img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        if self.augment:
            # Satellite patches have no "up", so flips and 90-degree turns are safe augmentations
            img = np.rot90(img, k=np.random.randint(4), axes=(1, 2))
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1]
        return torch.from_numpy(img.copy()), torch.tensor(label)


def band_stats(dataset):
    """Per-band mean/std over the training patches, used to normalise every split."""
    total, total_sq, n = 0.0, 0.0, 0
    for path, _ in dataset.items:
        img = load_patch(path).reshape(11, -1)
        total = total + img.sum(1)
        total_sq = total_sq + (img ** 2).sum(1)
        n += img.shape[1]
    mean = total / n
    std = np.sqrt(total_sq / n - mean ** 2) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)
