"""Stimulus reconstruction from population activity (experimental branch).

This is the inverse of the encoding analyses: given the population response
``Y`` (``n_bins x n_units`` spike counts), recover the sparse-noise stimulus
movie ``S`` (``n_bins x n_cells`` binary flashes).

Five algorithms spanning the main model families are implemented:

============= ===========================================================
``ridge``       linear, one weight per (cell, unit, lag) - the OLE baseline
``pls``         low-rank linear regression (partial least squares)
``kernel``      RBF kernel ridge - the nonlinear regressor
``position``    continuous (x, y) position regression, converted back into a
                gaussian cell score (few parameters, but only flash bins are
                informative, so it needs strong shrinkage)
``matched_filter`` encoding-estimated per-unit kernels applied as a matched
                filter - the classic dense read-out of a linear encoder
``softmax``     multinomial logistic over the 297 cells plus a no-flash class
``glm_map``     generative: inverts the Poisson encoding model by maximising
                the spike-train likelihood under an L1 / non-negativity prior
============= ===========================================================

Every decoder is fitted on a subset of the time bins and evaluated on held-out
*contiguous blocks* with a guard gap at least as wide as the feature window, so
no training bin shares a response window with a test bin.

Metrics are reconstruction-focused: detection AUC (did anything flash?),
top-1 / top-5 localisation accuracy and localisation error on flash bins,
per-cell correlation of the reconstructed movie, and global movie R2. Passing a
time-shifted target gives the chance level of every metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class ReconstructionConfig:
    """Parameters of the reconstruction experiment.

    ``pre`` / ``post`` define the response window (in bins) around the stimulus
    bin ``t``: features are the responses at ``t + lag - pre`` to
    ``t + lag + post``. ``post > 0`` makes the decoder non-causal by that many
    bins (offline reconstruction), which is the usual convention for stimulus
    reconstruction and is reported explicitly.
    """

    pre: int = 2                 # response bins before the stimulus bin
    post: int = 1                # response bins after it (non-causal look-ahead)
    lag: int = 1                 # stimulus -> response latency in bins
    n_folds: int = 5             # contiguous held-out time blocks
    gap: int = 4                 # bins dropped around every test block (>= pre+post)

    ridge_alpha: float = 10.0
    pls_components: int = 10

    position_alpha: float = 1000.0  # strong shrinkage: 2 targets, few flash bins
    position_sigma: float = 1.5    # gaussian width (grid units) of the cell score

    kernel_alpha: float = 1.0
    kernel_gamma: float | None = None   # None -> sklearn 'scale' heuristic
    kernel_max_train: int = 3000

    softmax_C: float = 0.1
    softmax_max_iter: int = 300
    softmax_max_train: int = 2500

    glm_alpha: float = 1.0         # L2 penalty of the encoding GLM
    glm_fit_max_iter: int = 200
    glm_l1: float = 0.05           # L1 / non-negativity prior of the MAP decode
    glm_ista_iter: int = 200

    def n_features(self, n_units: int) -> int:
        return int(n_units) * (self.pre + self.post + 1)


#: Decoder names accepted by :func:`cross_validate_reconstruction`.
METHODS = ("ridge", "pls", "kernel", "position", "matched_filter",
           "softmax", "glm_map")


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def lagged_features(Y, cfg: ReconstructionConfig | None = None):
    """Stack the response bins that can inform each stimulus bin.

    Row ``t`` of the result holds ``Y[t + lag - pre : t + lag + post + 1]``
    flattened, i.e. the population response at the stimulus-to-response latency
    ``lag`` and its neighbours. Rows whose window falls outside the recording
    are marked invalid and excluded from fitting and evaluation.

    Returns
    -------
    X : ndarray (n_bins, n_units * (pre + post + 1))
    valid : ndarray of bool (n_bins,)
    """
    cfg = cfg or ReconstructionConfig()
    Y = np.asarray(Y, float)
    n_bins, n_units = Y.shape
    offsets = np.arange(-cfg.pre, cfg.post + 1)
    idx = (np.arange(n_bins)[:, None] + cfg.lag) + offsets[None, :]
    valid = ((idx >= 0) & (idx < n_bins)).all(axis=1)

    X = np.zeros((n_bins, len(offsets) * n_units))
    rows = np.flatnonzero(valid)
    for j, _ in enumerate(offsets):
        src = np.clip(idx[rows, j], 0, n_bins - 1)
        X[rows, j * n_units:(j + 1) * n_units] = Y[src]
    return X, valid


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------
def cell_centres(grid):
    """Grid-cell id -> (x, y) centre arrays, matching the cell indexing.

    ``reconstruct_sparse_noise`` numbers cells ``y * n_x + x`` (x varying
    fastest, matching :meth:`SparseNoiseGrid.reshape_map`), so the y coordinate
    of cell ``c`` is ``ys[c // n_x]`` and its x coordinate ``xs[c % n_x]``.
    """
    return np.tile(grid.xs, grid.n_y), np.repeat(grid.ys, grid.n_x)


def map_centroid(scores, grid):
    """Centre of mass of each score map, in grid coordinates.

    Gives every decoder a continuous position read-out (not just the argmax
    cell), which is what the ``position_error`` metric uses.
    """
    scores = np.asarray(scores, float)
    x_c, y_c = cell_centres(grid)
    w = np.clip(scores, 0, None)
    total = w.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return np.column_stack([(w * x_c).sum(axis=1) / total[:, 0],
                            (w * y_c).sum(axis=1) / total[:, 0]])


def evaluate_reconstruction(scores, movie, grid, valid=None,
                            detection=None, positions=None) -> dict:
    """Reconstruction metrics for one decoder.

    Parameters
    ----------
    scores : ndarray (n_bins, n_cells)
        Continuous per-cell reconstruction (higher = more likely to flash).
    movie : ndarray (n_bins, n_cells) of bool
        True stimulus.
    valid : ndarray of bool, optional
        Rows to evaluate (default: all).
    detection : ndarray (n_bins,), optional
        Flash evidence per bin; defaults to the maximum over cells. A
        classifier can supply ``1 - P(no flash)`` here, which is better
        calibrated than the max of the per-cell probabilities.
    """
    scores = np.asarray(scores, float)
    S = np.asarray(movie, bool)
    if valid is None:
        valid = np.ones(len(S), bool)
    valid = np.asarray(valid, bool) & np.isfinite(scores).all(axis=1)
    sc, sv = scores[valid], S[valid]
    n_cells = S.shape[1]
    x_c, y_c = cell_centres(grid)

    out = {"n_eval_bins": int(valid.sum()),
           "n_flash_bins": int(sv.any(axis=1).sum()),
           "n_flash_cells": int(sv.sum())}

    # --- detection: did any cell flash in this bin? ---
    has_flash = sv.any(axis=1)
    det = np.asarray(detection, float)[valid] if detection is not None \
        else sc.max(axis=1)
    if 0 < has_flash.sum() < len(has_flash):
        out["detection_auc"] = float(roc_auc_score(has_flash, det))
    else:
        out["detection_auc"] = np.nan

    # --- localisation on bins that did contain a flash ---
    flash_bins = np.flatnonzero(has_flash)
    if len(flash_bins):
        top = np.argsort(-sc[flash_bins], axis=1)
        hit1 = sv[flash_bins][np.arange(len(flash_bins)), top[:, 0]]
        out["top1_accuracy"] = float(hit1.mean())
        k5 = min(5, n_cells)
        hits5 = np.take_along_axis(sv[flash_bins], top[:, :k5], axis=1).any(axis=1)
        out["top5_accuracy"] = float(hits5.mean())

        # distance from the predicted cell to the nearest true flash cell
        pred = top[:, 0]
        d = np.sqrt((x_c[pred][:, None] - x_c[None, :]) ** 2
                    + (y_c[pred][:, None] - y_c[None, :]) ** 2)
        true_mask = sv[flash_bins]
        d_true = np.where(true_mask, d, np.inf).min(axis=1)
        out["localisation_error"] = float(np.mean(d_true))
        out["top1_accuracy_chance"] = float(sv[flash_bins].mean(axis=0).mean())
    else:
        out.update({"top1_accuracy": np.nan, "top5_accuracy": np.nan,
                    "localisation_error": np.nan, "top1_accuracy_chance": np.nan})

    # continuous position read-out: distance from the estimated position (the
    # centre of mass of the score map) to the nearest true flash cell
    if positions is not None and len(flash_bins):
        pos = np.asarray(positions, float)[valid][flash_bins]
        dpos = np.sqrt((pos[:, [0]] - x_c[None, :]) ** 2
                       + (pos[:, [1]] - y_c[None, :]) ** 2)
        out["position_error"] = float(np.mean(dpos.min(axis=1)))
    else:
        out["position_error"] = np.nan

    # --- per-cell correlation and global movie R2 ---
    n_flash_per_cell = sv.sum(axis=0)
    usable = n_flash_per_cell >= 3
    if usable.any():
        a = sc[:, usable] - sc[:, usable].mean(axis=0, keepdims=True)
        b = sv[:, usable] - sv[:, usable].mean(axis=0, keepdims=True)
        denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(denom > 0, (a * b).sum(axis=0) / denom, np.nan)
        finite = np.isfinite(r)
        out["mean_cell_correlation"] = (float(r[finite].mean()) if finite.any()
                                        else np.nan)
        out["n_cells_evaluated"] = int(usable.sum())
    else:
        out["mean_cell_correlation"] = np.nan
        out["n_cells_evaluated"] = 0

    sst = ((sv - sv.mean()) ** 2).sum()
    out["movie_r2"] = float(1 - ((sv - sc) ** 2).sum() / sst) if sst > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Discriminative decoders
# ---------------------------------------------------------------------------
def _standardise(X_train, X_test):
    """Z-score features with training statistics only (no test leakage)."""
    mu = X_train.mean(axis=0)
    sd = np.where(X_train.std(axis=0) > 1e-9, X_train.std(axis=0), 1.0)
    return (X_train - mu) / sd, (X_test - mu) / sd


def decode_ridge(X_train, S_train, X_test, cfg):
    """Linear L2 regression: one weight per (cell, unit, lag)."""
    model = Ridge(alpha=cfg.ridge_alpha).fit(X_train, S_train)
    return model.predict(X_test), {"alpha": cfg.ridge_alpha}


def decode_pls(X_train, S_train, X_test, cfg):
    """Low-rank linear regression (partial least squares).

    Keeps the few stimulus dimensions the population actually carries, which
    regularises the 297-dimensional readout more explicitly than the ridge
    penalty does.
    """
    k = int(min(cfg.pls_components, X_train.shape[1], S_train.shape[1]))
    model = PLSRegression(n_components=k, scale=False).fit(X_train, S_train)
    return model.predict(X_test), {"n_components": k}


def decode_kernel(X_train, S_train, X_test, cfg, seed=0):
    """RBF kernel ridge: the nonlinear regressor, in dual (kernelised) form.

    The kernel width defaults to sklearn's ``scale`` heuristic
    ``1 / (n_features * var)``: on z-scored features that is ~1/n_features,
    which is the right order for a kernel that is neither an identity matrix
    (gamma too small) nor noise (gamma too large).
    """
    rng = np.random.default_rng(seed)
    if len(X_train) > cfg.kernel_max_train:
        idx = np.sort(rng.choice(len(X_train), cfg.kernel_max_train, replace=False))
        X_train, S_train = X_train[idx], S_train[idx]
    gamma = (cfg.kernel_gamma if cfg.kernel_gamma is not None
             else 1.0 / (X_train.shape[1] * X_train.var()))
    model = KernelRidge(kernel="rbf", alpha=cfg.kernel_alpha,
                        gamma=gamma).fit(X_train, S_train)
    return model.predict(X_test), {"gamma": float(gamma),
                                   "alpha": cfg.kernel_alpha,
                                   "n_train": int(len(X_train))}


def decode_softmax(X_train, S_train, X_test, cfg, seed=0):
    """Multinomial logistic over the 297 cells plus a no-flash class.

    One classifier gives both read-outs: localisation is the argmax over the
    cell classes, detection is ``1 - P(no flash)``. ``class_weight="balanced"``
    is essential here - a single cell flashes in well under 1% of bins, so an
    unweighted fit would never predict a cell.
    """
    n_cells = S_train.shape[1]
    labels = np.full(len(S_train), n_cells, dtype=int)      # no-flash class
    flashed = np.flatnonzero(S_train.any(axis=1))
    labels[flashed] = np.argmax(S_train[flashed], axis=1)   # ties -> first cell

    rng = np.random.default_rng(seed)
    if len(X_train) > cfg.softmax_max_train:
        idx = np.sort(rng.choice(len(X_train), cfg.softmax_max_train, replace=False))
        X_train, labels = X_train[idx], labels[idx]

    model = LogisticRegression(C=cfg.softmax_C, max_iter=cfg.softmax_max_iter,
                               class_weight="balanced").fit(X_train, labels)
    proba = model.predict_proba(X_test)

    scores = np.zeros((len(X_test), n_cells))
    detection = np.zeros(len(X_test))
    for col, cls in enumerate(model.classes_):
        if cls < n_cells:
            scores[:, int(cls)] = proba[:, col]
        else:
            detection = 1.0 - proba[:, col]
    return scores, {"detection": detection, "C": cfg.softmax_C,
                    "n_classes": int(len(model.classes_)),
                    "n_train": int(len(X_train))}


def decode_position(X_train, S_train, X_test, cfg, grid, seed=0):
    """Continuous position regression turned back into a cell score.

    Rather than estimating 297 independent cell filters (which is
    over-parameterised: each cell flashes only a dozen times per session), this
    predicts the flash's continuous grid position ``(x, y)`` with two ridge
    regressions and converts the prediction into cell scores with a Gaussian of
    width ``position_sigma``. The Gaussian is what makes the output comparable
    with the other decoders: it is a proper spatial read-out, and its peak can
    be used for localisation while its maximum serves as the detection score.
    """
    x_c, y_c = cell_centres(grid)
    cells = S_train.argmax(axis=1)
    has_flash = S_train.any(axis=1)
    target = np.column_stack([x_c[cells], y_c[cells]])
    model = Ridge(alpha=cfg.position_alpha).fit(X_train[has_flash],
                                                target[has_flash])
    pred = model.predict(X_test)
    d2 = (x_c[None, :] - pred[:, [0]]) ** 2 + (y_c[None, :] - pred[:, [1]]) ** 2
    scores = np.exp(-d2 / (2.0 * cfg.position_sigma ** 2))

    # The Gaussian score map is ~1 at its own peak by construction, so it cannot
    # serve as a detection statistic; a dedicated flash detector is fitted
    # instead (the two-stage "detect, then localise" decoder).
    if 0 < has_flash.sum() < len(has_flash):
        detector = LogisticRegression(C=cfg.softmax_C,
                                      max_iter=cfg.softmax_max_iter,
                                      class_weight="balanced")
        detector.fit(X_train, has_flash.astype(int))
        detection = detector.predict_proba(X_test)[:, 1]
    else:
        detection = scores.max(axis=1)
    return scores, {"detection": detection, "position_sigma": cfg.position_sigma,
                    "alpha": cfg.position_alpha,
                    "n_train": int(has_flash.sum())}


# ---------------------------------------------------------------------------
# Generative decoder: invert the fitted Poisson encoding model
# ---------------------------------------------------------------------------
def fit_encoding_model(Y, movie, bins, cfg: ReconstructionConfig):
    """Poisson encoding GLM fitted on ``bins`` only.

    Unit ``u``'s count at response bin ``r`` is modelled as
    ``lambda_u = exp(b_u + K_u . s(r - lag))`` - the stimulus-only part of the
    encoding model in :mod:`vision_pipeline.glm`. The spike-history term is
    omitted deliberately: when decoding we condition on the observed response,
    so history coefficients would only add nuisance parameters.

    Returns ``K`` (n_units, n_cells) and ``b`` (n_units,).
    """
    Y = np.asarray(Y, float)
    movie = np.asarray(movie, float)
    n_units, n_cells = Y.shape[1], movie.shape[1]
    bins = np.asarray(bins, int)
    src = bins - cfg.lag
    ok = (src >= 0) & (src < len(movie))
    design = movie[src[ok]]
    K = np.zeros((n_units, n_cells))
    b = np.zeros(n_units)
    for u in range(n_units):
        y = Y[bins[ok], u]
        if y.sum() < 5:
            b[u] = np.log(max(y.mean(), 1e-3))
            continue
        model = PoissonRegressor(alpha=cfg.glm_alpha,
                                 max_iter=cfg.glm_fit_max_iter).fit(design, y)
        K[u] = model.coef_
        b[u] = model.intercept_
    return K, b


def glm_map_decode(K, b, R, l1=0.05, n_iter=200, init=None, tol=1e-5):
    """MAP stimulus under the encoding model, by projected ISTA.

    Solves, for every row of the observed response ``R`` independently::

        s_hat = argmax_{s >= 0}  sum_u [y_u log lambda_u - lambda_u] - l1 * |s|_1
        lambda_u = exp(b_u + K_u . s)

    The smooth part is concave, so proximal gradient (ISTA) with
    soft-thresholding converges. The step is ``1 / (||K||_2^2 * max lambda)``
    and is halved whenever the objective would increase, which protects against
    the Lipschitz estimate going stale as the optimiser explores larger rates.
    """
    K = np.asarray(K, float)
    R = np.asarray(R, float)
    lipschitz_base = float(np.linalg.norm(K, 2) ** 2)
    S = (np.zeros((len(R), K.shape[1])) if init is None
         else np.maximum(np.asarray(init, float), 0.0))

    def objective(S):
        lam = np.exp(np.clip(S @ K.T + b, -30, 30))
        ll = np.where(R > 0, R * np.log(np.maximum(lam, 1e-12)), 0.0) - lam
        return float(-ll.sum() + l1 * np.abs(S).sum())

    obj = objective(S)
    for _ in range(int(n_iter)):
        lam = np.exp(np.clip(S @ K.T + b, -30, 30))
        grad = (R - lam) @ K                       # d(loglik)/ds
        step = 1.0 / max(lipschitz_base * float(lam.max()), 1e-12)
        cand, cand_obj = None, None
        for _ in range(12):
            cand = np.maximum(S + step * grad - step * l1, 0.0)
            cand_obj = objective(cand)
            if cand_obj <= obj:
                break
            step /= 2.0
        if cand_obj is None or cand_obj > obj:
            break                                # no improving step found
        delta = float(np.abs(cand - S).max())
        S, obj = cand, cand_obj
        if delta < tol:
            break
    return S


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------
def contiguous_folds(n_bins, cfg: ReconstructionConfig):
    """Contiguous test blocks, each surrounded by a guard gap.

    Splitting time into blocks (rather than shuffling bins) respects the slow
    drift of population activity; the gap of ``cfg.gap`` bins on both sides of
    every test block ensures no training bin shares a response window with a
    test bin, which a shuffled split would leak.
    """
    bounds = np.linspace(0, n_bins, cfg.n_folds + 1).astype(int)
    all_bins = np.arange(n_bins)
    folds = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        held = np.arange(max(lo - cfg.gap, 0), min(hi + cfg.gap, n_bins))
        folds.append((np.setdiff1d(all_bins, held), np.arange(lo, hi)))
    return folds


def _fold_predictions(method, X, movie, Y, cfg, train, test, fold=0, seed=0,
                      grid=None):
    """Fit one decoder on ``train`` bins and score ``test`` bins."""
    if method in ("glm_map", "matched_filter"):
        K, b = fit_encoding_model(Y, movie, train, cfg)
        R = Y[np.asarray(test, int) + cfg.lag]
        if method == "matched_filter":
            # Dense matched filter: the response projected onto the estimated
            # per-unit kernels. Every training bin contributes to estimating
            # the kernels, which is why it generalises far better than a
            # 297-way read-out fitted on flash bins alone.
            sc = R @ K
            return sc, sc.max(axis=1), {"n_units_glm": int(Y.shape[1])}
        sc = glm_map_decode(K, b, R, l1=cfg.glm_l1, n_iter=cfg.glm_ista_iter)
        return sc, sc.max(axis=1), {"n_units_glm": int(Y.shape[1])}

    X_train, X_test = _standardise(X[train], X[test])
    S_train = movie[train].astype(float)
    if method == "ridge":
        sc, meta = decode_ridge(X_train, S_train, X_test, cfg)
    elif method == "pls":
        sc, meta = decode_pls(X_train, S_train, X_test, cfg)
    elif method == "kernel":
        sc, meta = decode_kernel(X_train, S_train, X_test, cfg, seed=seed + fold)
    elif method == "position":
        sc, meta = decode_position(X_train, S_train, X_test, cfg, grid,
                                   seed=seed + fold)
    elif method == "softmax":
        sc, meta = decode_softmax(X_train, S_train, X_test, cfg, seed=seed + fold)
    else:
        raise ValueError(f"unknown method: {method!r}; choose from {METHODS}")
    detection = meta.get("detection")
    return sc, (sc.max(axis=1) if detection is None else detection), meta


def cross_validate_reconstruction(Y, movie, grid, cfg=None, methods=METHODS,
                                  verbose=True, seed=0):
    """Fit every decoder on training blocks, score held-out blocks.

    Returns
    -------
    summary : DataFrame      one row per method (pooled over all test bins)
    folds : DataFrame        one row per (method, fold) with per-fold metrics
    predictions : dict       method -> {"scores": (n_bins, n_cells) scores with
                             NaN outside the held-out blocks, "detection": the
                             matching (n_bins,) flash evidence}
    """
    cfg = cfg or ReconstructionConfig()
    Y = np.asarray(Y, float)
    movie = np.asarray(movie, bool)
    if cfg.gap < cfg.pre + cfg.post:
        raise ValueError("cfg.gap must be >= pre + post to prevent leakage")

    X, valid = lagged_features(Y, cfg)
    fold_index = contiguous_folds(len(Y), cfg)

    summary_rows, fold_rows, predictions = [], [], {}
    for method in methods:
        scores = np.full(movie.shape, np.nan)
        detection = np.full(len(movie), np.nan)
        positions = np.full((len(movie), 2), np.nan)
        per_fold = []
        for k, (train, test) in enumerate(fold_index):
            train_bins = train[valid[train]]
            test_bins = test[valid[test]]
            if len(train_bins) < 100 or len(test_bins) == 0:
                continue
            sc, det, _ = _fold_predictions(method, X, movie, Y, cfg,
                                           train_bins, test_bins, k, seed,
                                           grid=grid)
            scores[test_bins] = sc
            detection[test_bins] = det
            fold_positions = map_centroid(sc, grid)
            positions[test_bins] = fold_positions
            row = {"method": method, "fold": k, "n_train": len(train_bins),
                   "n_test": len(test_bins)}
            row.update(evaluate_reconstruction(sc, movie[test_bins], grid,
                                               detection=det,
                                               positions=fold_positions))
            per_fold.append(row)

        if not per_fold:
            continue
        fold_df = pd.DataFrame(per_fold)
        pooled = evaluate_reconstruction(scores, movie, grid, valid=valid,
                                        detection=detection,
                                        positions=positions)
        row = {"method": method, "n_folds": len(fold_df)}
        row.update(pooled)
        for col in ("detection_auc", "top1_accuracy", "top5_accuracy",
                    "localisation_error", "position_error",
                    "mean_cell_correlation", "movie_r2"):
            row[col + "_fold_sd"] = (float(fold_df[col].std(ddof=1))
                                     if len(fold_df) > 1 else np.nan)
            row["n_train"] = int(fold_df["n_train"].mean())
        summary_rows.append(row)
        fold_rows.extend(per_fold)
        predictions[method] = {"scores": scores, "detection": detection,
                               "positions": positions}
        if verbose:
            print(f"    {method:8s} AUC {pooled['detection_auc']:.3f} | "
                  f"top1 {pooled['top1_accuracy']:.2%} | "
                  f"top5 {pooled['top5_accuracy']:.2%} | "
                  f"r {pooled['mean_cell_correlation']:.3f} | "
                  f"R2 {pooled['movie_r2']:.3f}")
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), predictions


def chance_control(Y, movie, grid, cfg=None, method="ridge", shift=None,
                   verbose=False, seed=0):
    """Chance level: the same decoder fitted against a time-shifted target.

    Rolling the movie by a large offset preserves its marginal statistics but
    destroys the response-stimulus relationship, so every metric should fall to
    its chance value (AUC 0.5, top-1 ~ 1/297).
    """
    cfg = cfg or ReconstructionConfig()
    shift = int(shift if shift is not None else len(movie) // 2 + 7)
    shifted = np.roll(np.asarray(movie, bool), shift, axis=0)
    summary, folds, _ = cross_validate_reconstruction(
        Y, shifted, grid, cfg, methods=(method,), verbose=verbose, seed=seed)
    for frame in (summary, folds):
        if len(frame):
            frame["method"] = str(method) + "_control"
    return summary, folds