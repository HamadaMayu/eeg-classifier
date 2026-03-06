"""Image feature extraction with EVA-CLIP."""

import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import open_clip


device = "cuda" if torch.cuda.is_available() else "cpu"


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, path


def build_eva_clip(model_name: str = "EVA02-L-14-336", pretrained: str = "merged2b_s6b_b61k"):
    eva, _, eva_preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    eva = eva.to(device).eval()
    for p in eva.parameters():
        p.requires_grad = False
    return eva, eva_preprocess


@torch.no_grad()
def extract_eva_clip_features(
    image_paths: List[str],
    eva,
    eva_preprocess,
    batch_size: int = 64,
    num_workers: int = 2,
    normalize: bool = True,
    save_path: str | None = None,
):
    ds = ImagePathDataset(image_paths, eva_preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_feats = []
    all_paths = []

    for x, paths in tqdm(loader, desc="Extract EVA-CLIP feats"):
        x = x.to(device, non_blocking=True)
        feats = eva.encode_image(x)

        if feats.ndim == 3:
            feats = feats.mean(dim=1)

        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        all_feats.append(feats.cpu())
        all_paths.extend(list(paths))

    all_feats = torch.cat(all_feats, dim=0).numpy()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.save(save_path, all_feats)
        np.save(save_path.replace(".npy", "_paths.npy"), np.array(all_paths))

    return all_feats, all_paths


def load_image_paths(txt_path: str, img_root: str) -> List[str]:
    paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel:
                paths.append(os.path.join(img_root, rel))
    return paths
