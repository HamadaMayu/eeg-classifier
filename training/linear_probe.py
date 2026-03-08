"""Linear probing on top of the contrastive backbone."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.eeg_conformer import EEGConformerContrastive


device = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR = os.environ.get("WORK_DIR", ".")
CKPT_PATH = os.environ.get("CKPT_PATH", "contrastive.pt")


class ThingsEEGDatasetCls(Dataset):
    def __init__(self, split: str, work_dir: str = WORK_DIR):
        super().__init__()
        assert split in ["train", "val"]
        base = os.path.join(work_dir, "data", split)
        self.X = torch.from_numpy(np.load(os.path.join(base, "eeg.npy"))).float()
        self.y = torch.from_numpy(np.load(os.path.join(base, "labels.npy"))).long()
        self.subj = torch.from_numpy(np.load(os.path.join(base, "subject_idxs.npy"))).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.subj[i]

    @property
    def num_channels(self):
        return self.X.shape[1]

    @property
    def seq_len(self):
        return self.X.shape[2]


@torch.no_grad()
def extract_feats(backbone, loader, device, normalize_feat=True):
    backbone.eval()
    feats, labels = [], []
    for X, y, subj in tqdm(loader, desc="Extract feats"):
        X = X.to(device, non_blocking=True)
        subj = subj.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, feat = backbone(X, subj, return_feat=True, normalize_feat=normalize_feat)
        feats.append(feat.detach().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


class LinearProbe(nn.Module):
    def __init__(self, in_dim=768, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class FeatDataset(Dataset):
    def __init__(self, feat, y):
        self.feat = feat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.feat[i], self.y[i]


criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def eval_probe(probe, loader):
    probe.eval()
    correct, total, total_loss = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = probe(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def train_probe_one_epoch(probe, loader, optimizer):
    probe.train()
    correct, total, total_loss = 0, 0, 0.0
    for x, y in tqdm(loader, desc="Train probe"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


def main():
    train_ds = ThingsEEGDatasetCls("train")
    val_ds = ThingsEEGDatasetCls("val")

    backbone = EEGConformerContrastive(
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

    ckpt = torch.load(CKPT_PATH, map_location=device)
    backbone.load_state_dict(ckpt["model"], strict=False)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    train_loader_feat = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    val_loader_feat = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    train_feat, train_y = extract_feats(backbone, train_loader_feat, device, normalize_feat=True)
    val_feat, val_y = extract_feats(backbone, val_loader_feat, device, normalize_feat=True)

    probe = LinearProbe(in_dim=train_feat.shape[1], num_classes=5).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)

    train_feat_loader = DataLoader(FeatDataset(train_feat, train_y), batch_size=1024, shuffle=True, num_workers=0)
    val_feat_loader = DataLoader(FeatDataset(val_feat, val_y), batch_size=1024, shuffle=False, num_workers=0)

    best_val_acc = 0.0
    for epoch in range(1, 21):
        tr_loss, tr_acc = train_probe_one_epoch(probe, train_feat_loader, optimizer)
        va_loss, va_acc = eval_probe(probe, val_feat_loader)
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"probe": probe.state_dict()}, "best_linear_probe.pt")
            print("saved best_linear_probe.pt")


if __name__ == "__main__":
    main()
