"""Grad-CAM heatmaps for test patches: which pixels pushed the model towards 'debris'?

Usage: python src/gradcam.py --data /path/to/MARIDA --ckpt results/best.pt --out results/gradcam.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import MaridaPatches, load_patch
from model import build_model


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x):
        self.model.zero_grad()
        logit = self.model(x).squeeze()
        logit.backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)  # average gradient per channel
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam.detach().cpu().numpy(), torch.sigmoid(logit).item()


def rgb(path):
    img = load_patch(path)[[3, 2, 1]]  # Sentinel-2 B4, B3, B2 = red, green, blue
    img = np.transpose(img, (1, 2, 0))
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="results/best.pt")
    ap.add_argument("--out", default="results/gradcam.png")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["model"])
    model.eval()
    cam = GradCAM(model, model.layer4)
    ds = MaridaPatches(args.data, "test", ckpt["mean"], ckpt["std"])
    picks = [i for i, (_, y) in enumerate(ds.items) if y == 1][: args.n]

    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.4))
    for col, i in enumerate(picks):
        x, _ = ds[i]
        heat, prob = cam(x.unsqueeze(0))
        axes[0, col].imshow(rgb(ds.items[i][0]))
        axes[0, col].set_title(f"debris patch, p={prob:.2f}", fontsize=9)
        axes[1, col].imshow(rgb(ds.items[i][0]))
        axes[1, col].imshow(heat, cmap="jet", alpha=0.45)
        for ax in axes[:, col]:
            ax.axis("off")
    axes[0, 0].set_ylabel("RGB")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
