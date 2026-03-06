"""Inference script for test-time prediction and submission export."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.eeg_conformer import EEGConformerContrastive
from model.subcenter_cosface import SubCenterCosFace, EEGConformer_SubCosFace


device = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR = os.environ.get("WORK_DIR", ".")
CKPT_PATH = os.environ.get("CKPT_PATH", "best_subcosface_eeg2clip_K6.pt")


class ThingsEEGDatasetCls(Dataset):
    def __init__(self, split: str, work_dir: str = WORK_DIR):
        super().__init__()
        assert split in ["test"]
        base = os.path.join(work_dir, "data", split)
        self.X = torch.from_numpy(np.load(os.path.join(base, "eeg.npy"))).float()
        self.subject_idxs = torch.from_numpy(np.load(os.path.join(base, "subject_idxs.npy"))).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self):
        return self.X.shape[1]

    @property
    def seq_len(self):
        return self.X.shape[2]


@torch.no_grad()
def predict_logits_subcenter(model, test_loader):
    model.eval()
    all_logits = []
    C = model.classifier.num_classes
    K = model.classifier.K

    for X, subj in tqdm(test_loader, desc="Test inference"):
        X = X.to(device, non_blocking=True)
        subj = subj.to(device, non_blocking=True)
        _, feat = model.backbone(X, subj, return_feat=True, normalize_feat=True)
        feat_n = F.normalize(feat, dim=-1)
        W_n = F.normalize(model.classifier.weight, dim=-1)
        cos_all = feat_n @ W_n.t()
        B = cos_all.size(0)
        cos_ck = cos_all.view(B, C, K)
        cos_c, _ = cos_ck.max(dim=2)
        s = model.classifier.scale if hasattr(model.classifier, "scale") else 1.0
        logits = cos_c * s
        all_logits.append(logits.detach().cpu())

    return torch.cat(all_logits, dim=0).numpy()


def main():
    test_ds = ThingsEEGDatasetCls("test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    backbone = EEGConformerContrastive(
        num_classes=5,
        in_channels=test_ds.num_channels,
        seq_len=test_ds.seq_len,
        d_model=256,
        d_ff=1024,
        nhead=4,
        num_layers=3,
        kernel_size=15,
        p_drop=0.3,
        num_subjects=10,
        clip_feat_dim=768,
    ).to(device)
    classifier = SubCenterCosFace(in_dim=768, num_classes=5, K=6, s=20.0, m=0.25, learn_scale=True).to(device)
    model = EEGConformer_SubCosFace(backbone, classifier).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    all_logits = predict_logits_subcenter(model, test_loader)
    np.save("submission.npy", all_logits)
    print("Saved submission.npy", all_logits.shape)


if __name__ == "__main__":
    main()
