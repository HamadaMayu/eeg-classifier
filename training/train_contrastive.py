"""Contrastive training script pieces split from the notebook."""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model.eeg_conformer import EEGConformerContrastive


device = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR = os.environ.get("WORK_DIR", ".")


class ThingsEEGDataset(Dataset):
    def __init__(self, split: str, use_img_features: bool = False, work_dir: str = WORK_DIR):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.split = split
        self.use_img_features = use_img_features

        base = os.path.join(work_dir, "data", split)
        self.X = torch.from_numpy(np.load(os.path.join(base, "eeg.npy"))).float()
        self.subject_idxs = torch.from_numpy(np.load(os.path.join(base, "subject_idxs.npy"))).long()

        if split in ["train", "val"]:
            self.y = torch.from_numpy(np.load(os.path.join(base, "labels.npy"))).long()
            if self.use_img_features:
                eva_name = "train_eva_feats.npy" if split == "train" else "val_eva_feats.npy"
                img_feats = np.load(os.path.join(base, eva_name))
                self.img_features = torch.from_numpy(img_feats).float()
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.split in ["train", "val"]:
            if self.use_img_features:
                return self.X[i], self.y[i], self.subject_idxs[i], self.img_features[i]
            return self.X[i], self.y[i], self.subject_idxs[i]
        return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self):
        return self.X.shape[1]

    @property
    def seq_len(self):
        return self.X.shape[2]


def safe_l2_normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def clip_contrastive_loss(eeg_feat, img_feat, temperature=0.1):
    eeg_feat = safe_l2_normalize(eeg_feat, dim=-1)
    img_feat = safe_l2_normalize(img_feat, dim=-1)

    logits = (eeg_feat @ img_feat.T) / temperature
    targets = torch.arange(eeg_feat.size(0), device=eeg_feat.device)
    loss_e2i = F.cross_entropy(logits, targets)
    loss_i2e = F.cross_entropy(logits.T, targets)
    return (loss_e2i + loss_i2e) / 2


def train_contrastive_one_epoch(model, loader, optimizer, scheduler, device, temperature=0.1, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for X, y, subj, img_feat in loader:
        X = X.to(device, non_blocking=True)
        subj = subj.to(device, non_blocking=True)
        img_feat = img_feat.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, eeg_feat = model(X, subj, return_feat=True)
        loss = clip_contrastive_loss(eeg_feat, img_feat, temperature)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_contrastive(model, loader, device, temperature=0.1):
    model.eval()
    total_loss = 0.0
    for X, y, subj, img_feat in tqdm(loader, desc="Val (contrastive)"):
        X = X.to(device, non_blocking=True)
        subj = subj.to(device, non_blocking=True)
        img_feat = img_feat.to(device, non_blocking=True)
        _, eeg_feat = model(X, subj, return_feat=True)
        loss = clip_contrastive_loss(eeg_feat, img_feat, temperature=temperature)
        total_loss += loss.item()
    return total_loss / len(loader)


def build_contrastive_model(train_ds):
    return EEGConformerContrastive(
        num_classes=5,
        in_channels=train_ds.num_channels,
        seq_len=train_ds.seq_len,
        d_model=256,
        d_ff=1024,
        nhead=4,
        num_layers=3,
        kernel_size=15,
        p_drop=0.1,
        num_subjects=10,
        clip_feat_dim=768,
    ).to(device)


def main():
    train_ds = ThingsEEGDataset("train", use_img_features=True)
    val_ds = ThingsEEGDataset("val", use_img_features=True)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    model = build_contrastive_model(train_ds)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    num_epochs = 20
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_contrastive_one_epoch(model, train_loader, optimizer, scheduler, device, temperature=0.05)
        val_loss = eval_contrastive(model, val_loader, device, temperature=0.05)
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, "best_contrastive.pt")
            print("saved best_contrastive.pt")


if __name__ == "__main__":
    main()
