"""Multimodality analysis using ERP / PSD / embedding features."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def make_erp_mean_features(X_nct, time_ms, windows_ms=[(0, 200), (200, 400)], ch_idx=None):
    X_use = X_nct if ch_idx is None else X_nct[:, ch_idx, :]
    feats = []
    for (a, b) in windows_ms:
        idx = (time_ms >= a) & (time_ms < b)
        feats.append(X_use[:, :, idx].mean(axis=2))
    return np.concatenate(feats, axis=1).astype(np.float32)


def bandpower_from_psd(freqs, psd, band):
    f1, f2 = band
    idx = (freqs >= f1) & (freqs < f2)
    if idx.sum() == 0:
        return np.zeros(psd.shape[:-1], dtype=np.float32)
    return np.trapz(psd[..., idx], freqs[idx], axis=-1).astype(np.float32)


def make_psd_band_features(
    X_nct,
    sfreq=100.0,
    nperseg=64,
    bands=[(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)],
    ch_idx=None,
    log=True,
):
    X_use = X_nct if ch_idx is None else X_nct[:, ch_idx, :]
    freqs, psd = welch(X_use, fs=sfreq, nperseg=nperseg, axis=-1)

    feat_list = []
    for band in bands:
        bp = bandpower_from_psd(freqs, psd, band)
        feat_list.append(bp)

    feats = np.stack(feat_list, axis=-1)
    feats = feats.reshape(feats.shape[0], -1)
    if log:
        feats = np.log(feats + 1e-8)
    return feats.astype(np.float32)


def make_index_per_class(labels, n_classes=5, max_per_class=None, seed=42):
    rng = np.random.RandomState(seed)
    idx_by_class = {}
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        if (max_per_class is not None) and (len(idx) > max_per_class):
            idx = rng.choice(idx, size=max_per_class, replace=False)
        idx_by_class[c] = np.sort(idx)
    return idx_by_class


def fit_global_scaler_pca(feats, pca_dim=50, seed=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feats)
    p = min(pca_dim, Xs.shape[1], max(2, Xs.shape[0] - 1))
    pca = PCA(n_components=p, random_state=seed)
    Z = pca.fit_transform(Xs)
    info = {
        "scaler": scaler,
        "pca": pca,
        "pca_explained": float(np.sum(pca.explained_variance_ratio_)),
    }
    return Z.astype(np.float32), info


def gmm_scores_per_class(
    Z,
    labels,
    subjects,
    idx_by_class,
    Ks=(1, 2, 3, 4, 5, 6),
    seed=42,
    covariance_type="full",
    reg_covar=1e-6,
    n_init=5,
    max_iter=300,
):
    out = {}
    for c, idx in idx_by_class.items():
        if len(idx) == 0:
            continue
        Xc = Z[idx]
        Sc = subjects[idx]
        bic_list, aic_list, sil_list = [], [], []

        for K in Ks:
            gmm = GaussianMixture(
                n_components=K,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                random_state=seed,
                n_init=n_init,
                max_iter=max_iter,
            )
            gmm.fit(Xc)
            bic_list.append(float(gmm.bic(Xc)))
            aic_list.append(float(gmm.aic(Xc)))
            if K >= 2:
                z = gmm.predict(Xc)
                sil_list.append(float(silhouette_score(Xc, z)) if len(np.unique(z)) >= 2 else float("nan"))
            else:
                sil_list.append(float("nan"))

        out[c] = {
            "N": len(idx),
            "Ks": list(Ks),
            "BIC": bic_list,
            "AIC": aic_list,
            "SIL": sil_list,
            "bestK_BIC": int(Ks[int(np.argmin(np.array(bic_list)))]),
            "bestK_AIC": int(Ks[int(np.argmin(np.array(aic_list)))]),
            "Xc": Xc,
            "Sc": Sc,
        }
    return out


def plot_scores(results, title_prefix=""):
    for c, d in results.items():
        Ks = d["Ks"]
        plt.figure(figsize=(6, 3))
        plt.plot(Ks, d["BIC"], marker="o", label="BIC")
        plt.plot(Ks, d["AIC"], marker="o", label="AIC")
        plt.xlabel("K")
        plt.ylabel("score (lower is better)")
        plt.title(f"{title_prefix} | class {c}")
        plt.grid(True)
        plt.legend()
        plt.show()
