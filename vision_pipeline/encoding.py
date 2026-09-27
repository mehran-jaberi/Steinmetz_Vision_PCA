"""Stimulus encoding: contrast tuning, per-neuron regression and decoding.

All inferential procedures are accompanied by a resampling null where that is
informative (see :mod:`vision_pipeline.validation`), because in-sample R² and
cross-validated accuracy are easy to over-interpret on small trial counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from .config import AnalysisConfig
from .responses import build_response_matrix
from .utils import bh_fdr, condition_sort_key

#: Features entered into the per-neuron linear model (intercept added in code).
REG_FEATURES = ["contrast_left", "contrast_right", "interaction"]


@dataclass
class TuningResult:
    tuning: pd.DataFrame          # condition, n_trials
    matrix: np.ndarray            # (n_conditions, n_units) mean response
    sem: np.ndarray               # (n_conditions, n_units)
    conditions: np.ndarray        # condition labels
    X: np.ndarray                 # (n_use, n_units) trial-level responses
    conds: np.ndarray             # (n_use,) condition per trial
    use: np.ndarray               # boolean trial mask
    unit_ids: np.ndarray

    @property
    def order(self):
        """Indices sorting conditions by (left, right) contrast."""
        return sorted(range(len(self.conditions)),
                      key=lambda i: condition_sort_key(self.conditions[i]))


def compute_contrast_tuning(session, unit_ids, grating_df,
                            cfg: AnalysisConfig | None = None) -> TuningResult:
    """Mean response per (left, right) contrast condition."""
    cfg = cfg or AnalysisConfig()
    use = grating_df["included"].values & (grating_df["abs_contrast"].values > 0)
    align = grating_df.loc[use, "onset"].values
    X = build_response_matrix(session, unit_ids, align, cfg.stim_window)

    conds = grating_df.loc[use, "condition"].values
    uniq = pd.unique(conds)

    matrix = np.full((len(uniq), len(unit_ids)), np.nan)
    sem = np.full((len(uniq), len(unit_ids)), np.nan)
    rows = []
    for i, cond in enumerate(uniq):
        m = conds == cond
        matrix[i] = X[m].mean(axis=0)
        sem[i] = X[m].std(axis=0, ddof=1) / np.sqrt(max(int(m.sum()), 1))
        rows.append({"condition": cond, "n_trials": int(m.sum())})

    return TuningResult(
        tuning=pd.DataFrame(rows), matrix=matrix, sem=sem, conditions=uniq,
        X=X, conds=conds, use=use, unit_ids=np.asarray(unit_ids, dtype=int),
    )


def _ols(A, y):
    """Least-squares fit returning coefficients and the fitted values."""
    betas, *_ = np.linalg.lstsq(A, y, rcond=None)
    return betas, A @ betas


def _kfold_r2(A, y, n_splits, seed):
    """Mean held-out R² (1 - SSE/SST) of a linear model over K folds.

    Rows are shuffled deterministically so that results do not depend on trial
    order. Returns NaN when there are too few trials for the requested splits.
    """
    n = len(y)
    if n < n_splits * (A.shape[1] + 1):
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ss_res, ss_tot, n_test = 0.0, 0.0, 0
    for train, test in kf.split(A):
        try:
            betas, _ = _ols(A[train], y[train])
        except np.linalg.LinAlgError:
            return np.nan
        pred = A[test] @ betas
        ss_res += np.sum((y[test] - pred) ** 2)
        ss_tot += np.sum((y[test] - y[test].mean()) ** 2)
        n_test += len(test)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def regress_responses_on_stimulus(session, unit_ids, grating_df, design,
                                  cfg: AnalysisConfig | None = None,
                                  X: np.ndarray | None = None,
                                  n_perm: int = 0, verbose=True):
    """Per-neuron OLS of the response on stimulus features.

    Model: ``response ~ 1 + contrast_left + contrast_right + interaction``.

    Reported quantities
    -------------------
    r2 : in-sample R² (comparable to the original pipeline)
    r2_cv : K-fold cross-validated R² (honest goodness-of-fit)
    p : F-test p-value, FDR-corrected (`significant`)
    p_perm : permutation p-value for R² (optional, ``n_perm > 0``)
    side_preference : β(right) − β(left)

    ``X`` can be supplied to reuse a response matrix computed earlier (the
    original pipeline computed it twice).
    """
    cfg = cfg or AnalysisConfig()
    use = grating_df["included"].values & (grating_df["abs_contrast"].values > 0)
    if X is None:
        X = build_response_matrix(session, unit_ids, grating_df.loc[use, "onset"].values,
                                  cfg.stim_window)

    feats = design.loc[use, REG_FEATURES].values
    A = np.column_stack([np.ones(len(feats)), feats])
    k = feats.shape[1]
    n = len(feats)
    dof = max(n - k - 1, 1)

    cols = ["intercept"] + list(REG_FEATURES)
    rng = np.random.default_rng(cfg.random_seed)
    perm_idx = [rng.permutation(n) for _ in range(n_perm)]

    rows = []
    for j, uid in enumerate(unit_ids):
        y = X[:, j]
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            rows.append(
                {"cluster_id": int(uid), "r2": np.nan, "r2_cv": np.nan,
                 "p": np.nan, "p_perm": np.nan, "side_preference": np.nan}
            )
            continue

        betas, yhat = _ols(A, y)
        r2 = 1 - np.sum((y - yhat) ** 2) / ss_tot
        F = (r2 / k) / ((1 - r2) / dof) if r2 < 1 else np.inf
        p = 1 - stats.f.cdf(F, k, dof) if np.isfinite(F) else 0.0

        row = {
            "cluster_id": int(uid),
            "r2": r2,
            "r2_cv": _kfold_r2(A, y, cfg.cv_folds, cfg.random_seed),
            "p": p,
            "p_perm": np.nan,
            "side_preference": betas[2] - betas[1],   # β(R) − β(L)
        }
        if n_perm > 0:
            null = np.empty(n_perm)
            for i, idx in enumerate(perm_idx):
                yp = y[idx]
                sst = np.sum((yp - yp.mean()) ** 2)
                _, yh = _ols(A, yp)
                null[i] = 1 - np.sum((yp - yh) ** 2) / max(sst, 1e-12)
            row["p_perm"] = (1 + np.sum(null >= r2)) / (n_perm + 1)
        for cname, b in zip(cols, betas):
            row[f"beta_{cname}"] = b
        rows.append(row)

    reg = pd.DataFrame(rows)
    reg["p_fdr"] = np.nan
    reg["significant"] = bh_fdr(reg["p"].fillna(1.0).values, alpha=cfg.fdr_alpha)
    reg["significant_perm"] = (
        bh_fdr(reg["p_perm"].fillna(1.0).values, alpha=cfg.fdr_alpha)
        if n_perm > 0 else False
    )
    if verbose:
        print(
            f"  Regression: {int(reg['significant'].sum())}/{len(reg)} units "
            f"significantly encode stimulus contrast (FDR p<{cfg.fdr_alpha}); "
            f"median R2 = {reg['r2'].median():.3f}, "
            f"median CV R2 = {reg['r2_cv'].median():.3f}"
        )
    return reg


def unilateral_mask(grating_df, use):
    """Mask (relative to ``use``) of trials with exactly one side stimulated."""
    cl = grating_df.loc[use, "contrast_left"].values
    cr = grating_df.loc[use, "contrast_right"].values
    return (cl == 0) | (cr == 0)


def decode_side(X, grating_df, use, cfg: AnalysisConfig | None = None,
                return_scores=False):
    """Cross-validated logistic regression decoding of the stimulated side.

    ``X`` rows must be aligned to the ``use`` mask of ``grating_df``. Only
    unilateral trials (exactly one side stimulated) are decoded, and the label
    is "was the right side stimulated?".

    Returns
    -------
    (mean_accuracy, std_accuracy, n_trials, labels, X_decoded) or with
    ``return_scores=True`` also the per-fold scores.
    """
    cfg = cfg or AnalysisConfig()
    mask = unilateral_mask(grating_df, use)
    y = (grating_df.loc[use, "contrast_right"].values[mask] > 0).astype(int)
    Xd = X[mask]

    n_min_class = min(np.bincount(y).min(), len(y))
    folds = int(min(cfg.cv_folds, n_min_class))
    if folds < 2:
        return (np.nan, np.nan, len(y), y, Xd) if not return_scores else \
            (np.nan, np.nan, len(y), y, Xd, np.array([]))

    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.random_seed)
    scores = cross_val_score(clf, Xd, y, cv=cv, scoring="accuracy")
    if return_scores:
        return scores.mean(), scores.std(), len(y), y, Xd, scores
    return scores.mean(), scores.std(), len(y), y, Xd


def decode_side_null(Xd, y, cfg: AnalysisConfig | None = None,
                     n_perm: int | None = None):
    """Label-permutation null for :func:`decode_side`.

    Returns the mean and 95th percentile of the null accuracy distribution, so
    decoding performance can be compared against chance rather than assumed to
    be meaningful.
    """
    cfg = cfg or AnalysisConfig()
    n_perm = cfg.n_permutations if n_perm is None else n_perm
    rng = np.random.default_rng(cfg.random_seed)

    n_min_class = min(np.bincount(y).min(), len(y))
    folds = int(min(cfg.cv_folds, n_min_class))
    accs = np.full(n_perm, np.nan)
    if folds < 2:
        return np.nan, np.nan, accs

    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    for i in range(n_perm):
        yp = rng.permutation(y)
        cv = StratifiedKFold(n_splits=folds, shuffle=True,
                             random_state=int(rng.integers(1 << 31)))
        accs[i] = cross_val_score(clf, Xd, yp, cv=cv, scoring="accuracy").mean()
    return float(np.nanmean(accs)), float(np.nanpercentile(accs, 95)), accs


def decode_choice(session, unit_ids, grating_df, cfg: AnalysisConfig | None = None):
    """Cross-validated decoding of the animal's choice from population activity.

    Included for completeness: it mirrors the Part-1 choice analysis at the
    population level. Only trials with a behavioural response (left/right) are
    used, and the label is "right choice".
    """
    cfg = cfg or AnalysisConfig()
    tr = session.trial_info
    choice = np.asarray(tr["response_choice"], float)
    if len(choice) != len(grating_df):
        raise ValueError("choice and grating tables are misaligned")

    valid = np.isin(choice, (-1.0, 1.0)) & grating_df["included"].values
    align = grating_df.loc[valid, "onset"].values
    X = build_response_matrix(session, unit_ids, align, cfg.stim_window)
    y = (choice[valid] > 0).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, len(y)
    folds = int(min(cfg.cv_folds, np.bincount(y).min()))
    if folds < 2:
        return np.nan, np.nan, len(y)
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.random_seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std()), int(len(y))