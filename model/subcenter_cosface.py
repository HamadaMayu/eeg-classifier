"""Sub-center CosFace classifier modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterCosFace(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        K: int = 3,
        s: float = 20.0,
        m: float = 0.35,
        learn_scale: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.K = K
        self.m = float(m)
        self.weight = nn.Parameter(torch.randn(num_classes * K, in_dim))
        nn.init.xavier_uniform_(self.weight)

        if learn_scale:
            self.scale = nn.Parameter(torch.tensor(float(s)))
        else:
            self.register_buffer("scale", torch.tensor(float(s)))

    def forward(self, x: torch.Tensor, y=None, apply_margin: bool = False) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        W = F.normalize(self.weight, dim=-1)
        cos_all = x @ W.t()
        B = x.size(0)
        cos_ck = cos_all.view(B, self.num_classes, self.K)
        cos_c, _ = cos_ck.max(dim=2)

        if apply_margin:
            if y is None:
                raise ValueError("y is required when apply_margin=True")
            y = y.long()
            cos_c = cos_c.clone()
            cos_c[torch.arange(B, device=x.device), y] -= self.m

        logits = cos_c * self.scale
        return logits


class EEGConformer_SubCosFace(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: SubCenterCosFace):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, X: torch.Tensor, subj: torch.Tensor, y=None, apply_margin: bool = False) -> torch.Tensor:
        _, feat = self.backbone(X, subj, return_feat=True, normalize_feat=True)
        logits = self.classifier(feat, y=y, apply_margin=apply_margin)
        return logits
