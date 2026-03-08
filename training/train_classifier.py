"""Final classification training with Sub-center CosFace."""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.eeg_conformer import EEGConformerContrastive
from model.subcenter_cosface import SubCenterCosFace, EEGConformer_SubCosFace


device = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR = os.environ.get("WORK_DIR", ".")
CKPT_PATH = os.environ.get("CKPT_PATH", "contrastive.pt")


class ThingsEEGDatasetCls(Dataset):
    def __init__(self, split: str, work_dir: str = WORK_DIR):
        super().__init__()
        assert split in ["train", "val", "test"]
        base = os.path.join(work_dir, "data", split)
        self.X = torch.from_numpy(np.load(os.path.join(base, "eeg.npy"))).float()
        self.subject_idxs = torch.from_numpy(np.load(os.path.join(base, "subject_idxs.npy"))).long()
        self.y = torch.from_numpy(np.load(os.path.join(base, "labels.npy"))).long() if split in ["train", "val"] else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i], self.subject_idxs[i]
        return self.X[i], self.y[i], self.subject_idxs[i]

    @property
    def num_channels(self):
        return self.X.shape[1]

    @property
    def seq_len(self):
        return self.X.shape[2]


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for X, y, subj in tqdm(loader, desc="Val (subcosface)"):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        subj = subj.to(device, non_blocking=True)
        logits = model(X, subj, y=None, apply_margin=False)
        loss = ce(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def train_epoch(model, loader, optimizer, scheduler=None, grad_clip=1.0):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for X, y, subj in tqdm(loader, desc="Train (subcosface)"):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        subj = subj.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X, subj, y=y, apply_margin=True)
        loss = ce(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def main():
    train_ds = ThingsEEGDatasetCls("train")
    val_ds = ThingsEEGDatasetCls("val")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    backbone = EEGConformerContrastive(
        num_classes=5,
        in_channels=train_ds.num_channels,
        seq_len=train_ds.seq_len,
        d_model=256,
        d_ff=1024,
        nhead=4,
        num_layers=3,
        kernel_size=15,
        p_drop=0.3,
        num_subjects=10,
        clip_feat_dim=768,
    ).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    backbone.load_state_dict(ckpt["model"], strict=False)

    for p in backbone.parameters():
        p.requires_grad = False
    for p in backbone.eeg2clip.parameters():
        p.requires_grad = True

    classifier = SubCenterCosFace(in_dim=768, num_classes=5, K=6, s=20.0, m=0.25, learn_scale=True).to(device)
    model = EEGConformer_SubCosFace(backbone, classifier).to(device)

    params = [
        {"params": backbone.eeg2clip.parameters(), "lr": 3e-4},
        {"params": classifier.parameters(), "lr": 3e-4},
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-2)

    num_epochs = 50
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc = 0.0
    save_path = "subcosface.pt"

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler=scheduler, grad_clip=1.0)
        va_loss, va_acc = eval_epoch(model, val_loader)
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict()}, save_path)
            print(f"saved {save_path}")


if __name__ == "__main__":
    main()
