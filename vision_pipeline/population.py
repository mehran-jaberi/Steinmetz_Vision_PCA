"""Population-level dimensionality reduction (PCA) and condition structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import AnalysisConfig
from .responses import binned_response_tensor, condition_average


@dataclass
class PCAResult:
    pca: PCA
    scaler: StandardScaler
    scores: np.ndarray
    evr: np.ndarray
    cum_evr: np.ndarray
    within_mean: float
    across_mean: float
    within_across_ratio: float
    discriminability: float

    def component_weights(self, unit_ids, n=3):
        """Loading table for the first ``n`` components."""
        cols = [f"PC{i + 1}" for i in range(min(n, self.pca.components_.shape[0]))]
        return pd.DataFrame(self.pca.components_[: len(cols)].T,
                            columns=cols, index=list(unit_ids))


def population_pca(X, conditions, cfg: AnalysisConfig | None = None,
                   n_dims_for_distance: int = 3) -> PCAResult:
    """PCA on the trial x unit response matrix, plus condition separability.

    ``X`` is z-scored per unit before PCA so that high-rate units do not
    dominate. Condition structure is summarised by the ratio of the mean
    within-condition to mean across-condition distance in the leading
    ``n_dims_for_distance`` dimensions; ``discriminability = 1 - ratio``.
    """
    cfg = cfg or AnalysisConfig()
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("population_pca expects a (trials, units) matrix "
                         "with at least two trials")

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=min(cfg.n_pca_components, Xz.shape[1], Xz.shape[0]))
    scores = pca.fit_transform(Xz)
    evr = pca.explained_variance_ratio_

    k = min(n_dims_for_distance, scores.shape[1])
    s = scores[:, :k]
    conds = np.asarray(conditions)
    uniq = list(pd.unique(conds))

    within = []
    for c in uniq:
        m = conds == c
        if m.sum() < 2:
            continue
        d = np.linalg.norm(s[m][:, None, :] - s[m][None, :, :], axis=2)
        within.append(d[np.triu_indices(int(m.sum()), 1)].mean())

    across = []
    for i, c1 in enumerate(uniq):
        m1 = conds == c1
        for c2 in uniq[i + 1:]:
            m2 = conds == c2
            if m1.sum() and m2.sum():
                d = np.linalg.norm(s[m1][:, None, :] - s[m2][None, :, :], axis=2)
                across.append(d.mean())

    ratio = (np.mean(within) / np.mean(across)) if within and across else np.nan
    return PCAResult(
        pca=pca, scaler=scaler, scores=scores, evr=evr, cum_evr=np.cumsum(evr),
        within_mean=float(np.mean(within)) if within else np.nan,
        across_mean=float(np.mean(across)) if across else np.nan,
        within_across_ratio=float(ratio),
        discriminability=float(1 - ratio) if np.isfinite(ratio) else np.nan,
    )


def time_resolved_pca(session, unit_ids, align_times, conditions,
                      cfg: AnalysisConfig | None = None, n_components=3):
    """Condition-averaged population trajectories projected into PC space.

    Returns
    -------
    bin_centers : ndarray (n_bins,)
    traj : ndarray (n_conditions, n_bins, n_components) condition trajectories
    evr : ndarray explained variance of the trajectory PCA
    labels : list of condition labels
    tensor : ndarray (n_trials, n_bins, n_units) trial-level counts
    """
    cfg = cfg or AnalysisConfig()
    edges, tensor = binned_response_tensor(
        session, unit_ids, align_times, cfg.traj_window, cfg.traj_bin
    )
    labels, means, _ = condition_average(tensor, conditions,
                                         cfg.min_condition_trials)
    if not labels:
        raise ValueError("no condition has enough trials for a trajectory")

    n_tbins = means.shape[1]
    flat = means.reshape(-1, means.shape[2])
    flat_z = StandardScaler().fit_transform(flat)
    pca = PCA(n_components=min(n_components, flat_z.shape[1], flat_z.shape[0]))
    proj = pca.fit_transform(flat_z).reshape(len(labels), n_tbins, -1)

    centers = edges[:-1] + cfg.traj_bin / 2
    return centers, proj, pca.explained_variance_ratio_, labels, tensor