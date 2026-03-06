"""EEG visualization utilities split from the notebook."""

import os
import numpy as np
import matplotlib.pyplot as plt


CH_NAMES = ['Pz','P3','P7','O1','Oz','O2','P4','P8','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2']
START_MS = -200
STEP_MS = 10


def build_time_axis(T: int, start_ms: int = START_MS, step_ms: int = STEP_MS):
    return np.arange(T) * step_ms + start_ms


def plot_representative_erp(X: np.ndarray, ch_names=None):
    ch_names = CH_NAMES if ch_names is None else ch_names
    time = build_time_axis(X.shape[2])
    erp = X.mean(axis=0)

    plt.figure(figsize=(9, 4))
    for ch in ["O1", "Oz", "O2", "P6", "P7", "P8", "PO7", "PO8"]:
        i = ch_names.index(ch)
        plt.plot(time, erp[i], label=ch)
    plt.axvline(0, color="k", linestyle="--", alpha=0.4)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("ERP (representative channels)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_top5_channels(X: np.ndarray, ch_names=None):
    ch_names = CH_NAMES if ch_names is None else ch_names
    time = build_time_axis(X.shape[2])
    erp = X.mean(axis=0)
    ptp = erp.max(axis=1) - erp.min(axis=1)
    top5_idx = np.argsort(ptp)[::-1][:5]

    plt.figure(figsize=(9, 4))
    for i in top5_idx:
        plt.plot(time, erp[i], label=ch_names[i])
    plt.axvline(0, color="k", linestyle="--", alpha=0.4)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("ERP (Top-5 channels)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_classwise_channel(X: np.ndarray, labels: np.ndarray, channel: str, ch_names=None, n_classes: int = 5):
    ch_names = CH_NAMES if ch_names is None else ch_names
    time = build_time_axis(X.shape[2])
    i = ch_names.index(channel)
    plt.figure(figsize=(9, 4))
    for c in range(n_classes):
        erp_c = X[labels == c].mean(axis=0)
        plt.plot(time, erp_c[i], label=f"class {c}")
    plt.axvline(0, color="k", linestyle="--", alpha=0.4)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title(f"ERP {channel} (class-wise)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_subject_overlay_for_channel(X, subj, ch, ch_names=None):
    ch_names = CH_NAMES if ch_names is None else ch_names
    time = build_time_axis(X.shape[2])
    c = ch_names.index(ch)
    subjects = np.unique(subj)

    plt.figure(figsize=(14, 4))
    for s in subjects:
        idx = subj == s
        erp_s = X[idx, c, :].mean(axis=0)
        plt.plot(time, erp_s, label=f"subj{s}")

    plt.axvline(0, color="k", linestyle="--", alpha=0.4)
    plt.title(f"{ch} (ERP by subject)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend(ncol=5, fontsize=8)
    plt.grid(True)
    plt.show()


def load_train_arrays(work_dir: str):
    base = os.path.join(work_dir, "data", "train")
    X = np.load(os.path.join(base, "eeg.npy"))
    y = np.load(os.path.join(base, "labels.npy"))
    subj = np.load(os.path.join(base, "subject_idxs.npy"))
    return X, y, subj
