"""EEG Conformer backbone modules.

Split from the original notebook-based implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectBlock(nn.Module):
    """Subject-specific feature alignment before the Conformer encoder."""

    def __init__(self, d_model: int, num_subjects: int, p_drop: float = 0.0):
        super().__init__()
        self.shared = nn.Linear(d_model, d_model, bias=True)
        self.W = nn.Parameter(torch.randn(num_subjects, d_model, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(num_subjects, d_model))
        self.dropout = nn.Dropout(p_drop)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, subj: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        W = self.W[subj]
        b = self.b[subj]
        x = torch.einsum("btd,bdh->bth", x, W) + b.unsqueeze(1)
        x = self.dropout(x)
        x = self.norm(x)
        return x


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w(x).squeeze(-1)
        a = torch.softmax(a, dim=1)
        feat = (x * a.unsqueeze(-1)).sum(dim=1)
        return feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p_drop: float = 0.4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_drop)
        self.activation = nn.SiLU() if hasattr(nn, "SiLU") else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, nhead: int, p_drop: float = 0.4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=p_drop,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return self.dropout(out)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 15, p_drop: float = 0.4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU() if hasattr(nn, "SiLU") else nn.ReLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, nhead: int, kernel_size: int = 15, p_drop: float = 0.1):
        super().__init__()
        self.ffn_scale = 0.5
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn1 = FeedForwardModule(d_model, d_ff, p_drop)
        self.ffn2 = FeedForwardModule(d_model, d_ff, p_drop)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttentionModule(d_model, nhead, p_drop)
        self.conv_module = ConformerConvModule(d_model, kernel_size, p_drop)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        x = x + self.ffn_scale * self.ffn1(self.ffn1_norm(x))
        x = x + self.self_attn(self.self_attn_norm(x), key_padding_mask=key_padding_mask)
        x = x + self.conv_module(x)
        x = x + self.ffn_scale * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class EEGConformerContrastive(nn.Module):
    """Backbone used for both contrastive training and downstream classification."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        seq_len: int,
        d_model: int = 128,
        d_ff: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        kernel_size: int = 15,
        p_drop: float = 0.2,
        num_subjects: int = 10,
        clip_feat_dim: int = 768,
    ):
        super().__init__()
        self.attn_pool = AttnPool(d_model)
        self.input_proj = nn.Linear(in_channels, d_model)
        self.subject_block = SubjectBlock(d_model=d_model, num_subjects=num_subjects, p_drop=0.3)

        core_ch = [4, 2, 7, 10, 14]  # Oz, P7, P8, PO7, PO8
        mid_ch = [11, 13, 5, 6]      # PO3, PO4, O1, O2
        channel_mask = torch.full((in_channels,), 0.4, dtype=torch.float32)
        channel_mask[mid_ch] = 0.7
        channel_mask[core_ch] = 1.0
        self.register_buffer("channel_mask", channel_mask.clamp(0.0, 1.0))

        self.subject_emb = nn.Embedding(num_subjects, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, d_ff, nhead, kernel_size, p_drop)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p_drop)
        self.eeg2clip = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, clip_feat_dim),
        )
        self.classifier = nn.Linear(clip_feat_dim, num_classes)

    def forward(
        self,
        X: torch.Tensor,
        subject_idxs: torch.Tensor,
        key_padding_mask=None,
        return_feat: bool = False,
        normalize_feat: bool = True,
    ):
        X = X.permute(0, 2, 1)
        X = X * self.channel_mask[None, None, :]
        X = self.input_proj(X)

        subject_idxs = subject_idxs.long() - 1
        X = self.subject_block(X, subject_idxs)
        subj = self.subject_emb(subject_idxs).unsqueeze(1)
        X = self.pos_enc(X + subj)

        for layer in self.layers:
            X = layer(X, key_padding_mask=key_padding_mask)

        X = self.attn_pool(X)
        X = self.dropout(X)
        feat = self.eeg2clip(X)
        if normalize_feat:
            feat = F.normalize(feat, dim=-1)

        logits = self.classifier(feat)
        return (logits, feat) if return_feat else logits
