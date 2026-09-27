"""Model-based population analysis 2/2: latent dynamics.

Three complementary latent-variable models are implemented:

``reduced_rank_regression``
    Supervised low-rank regression of the population time series on the
    stimulus movie. It answers "how many stimulus dimensions does the
    population use?" and is the same family of model used by the original
    MATLAB ``reducedRankRegression`` code shipped with the dataset.

``factor_analysis``
    Static Gaussian latent-variable model of the population time series. It
    estimates how much of each unit's variance is shared (population-level)
    rather than private.

``dynamic_factor`` / ``fit_var1``
    Dynamical models of the latent state. ``dynamic_factor`` fits a
    linear-Gaussian state-space model (AR(1) factors) by maximum likelihood via
    :mod:`statsmodels`; ``fit_var1`` fits a first-order vector autoregression to
    PCA latents and reports its eigenvalues, which reveal the decay rate and
    oscillation frequency of the population dynamics.

All three are fitted to binned, condition-averaged or trial-level population
activity computed elsewhere in the package; nothing here depends on plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from .config import AnalysisConfig


# ---------------------------------------------------------------------------
# Reduced-rank regression
# ---------------------------------------------------------------------------
def _inv_sqrt(M, tol=1e-10):
    """Inverse symmetric square root of a PSD matrix."""
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, tol, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def _sqrt(M, tol=1e-10):
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, tol, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def reduced_rank_regression(Y, X, rank):
    """Reduced-rank regression of multivariate ``Y`` on ``X``.

    Solves ``min_B ||Y - X B||_F`` subject to ``rank(B) <= rank``.

    Derivation (Izenman, 1975). With centred ``X``, ``Y`` the objective is
    ``tr(Y'Y) − 2 tr(B'Sxy) + tr(B'SxxB)``. Substituting
    ``B = Sxx^{-1/2} T`` (which leaves the rank unchanged) gives
    ``||T − Sxx^{-1/2} Sxy||_F²``, so the optimal rank-``r`` ``T`` is the
    truncated SVD of the whitened cross-covariance ``Sxx^{-1/2} Sxy`` and::

        B_r = Sxx^{-1/2} U_r S_r V_r'

    Note there is *no* ``Syy`` factor in ``B`` — including ``Syy^{-1/2}``
    (a plausible-looking variant) rescales the responses and destroys the fit.
    Extracting the SVD from ``Sxx^{-1/2} Sxy`` directly avoids that trap.

    Parameters
    ----------
    Y : ndarray (n, q)  responses (population activity)
    X : ndarray (n, p)  predictors (stimulus features)
    rank : int

    Returns
    -------
    dict with ``B``, ``intercept``, ``sigma`` (singular values), ``rank``.
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    n, q = Y.shape
    p = X.shape[1]
    rank = int(min(max(rank, 1), min(p, q)))

    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - x_mean, Y - y_mean
    Sxx = Xc.T @ Xc
    Sxy = Xc.T @ Yc

    Sxx_is = _inv_sqrt(Sxx)
    U, s, Vt = np.linalg.svd(Sxx_is @ Sxy, full_matrices=False)

    Ur, Sr, Vr = U[:, :rank], s[:rank], Vt[:rank].T
    B = Sxx_is @ Ur @ np.diag(Sr) @ Vr.T
    intercept = y_mean - x_mean @ B
    return {"B": B, "intercept": intercept, "sigma": s, "rank": rank}


def rrr_predict(fit, X):
    return np.asarray(X, float) @ fit["B"] + fit["intercept"]


def rrr_score(Y, X, fit):
    """R² of a reduced-rank fit (1 - SSE/SST over all entries)."""
    Yhat = rrr_predict(fit, X)
    sst = np.sum((Y - Y.mean(axis=0)) ** 2)
    if sst <= 0:
        return np.nan
    return float(1 - np.sum((Y - Yhat) ** 2) / sst)


def rrr_rank_curve(Y, X, ranks=None, cfg: AnalysisConfig | None = None,
                   n_splits=5):
    """In-sample and cross-validated R² as a function of the rank of ``B``.

    A cross-validated R² that stops improving marks the useful rank, i.e. the
    number of stimulus dimensions the population actually represents.
    """
    cfg = cfg or AnalysisConfig()
    ranks = ranks or cfg.rrr_ranks
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    n = len(Y)

    rng = np.random.default_rng(cfg.random_seed)
    folds = int(min(n_splits, max(n // 20, 2)))
    order = rng.permutation(n) if folds >= 2 else np.arange(n)
    splits = np.array_split(order, folds) if folds >= 2 else [order]

    rows = []
    for r in ranks:
        fit = reduced_rank_regression(Y, X, r)
        row = {"rank": int(r), "r2": rrr_score(Y, X, fit)}
        if folds >= 2:
            sse, sst = 0.0, 0.0
            for test in splits:
                train = np.setdiff1d(order, test, assume_unique=False)
                f = reduced_rank_regression(Y[train], X[train], r)
                pred = rrr_predict(f, X[test])
                sse += np.sum((Y[test] - pred) ** 2)
                sst += np.sum((Y[test] - Y[train].mean(axis=0)) ** 2)
            row["r2_cv"] = 1 - sse / sst if sst > 0 else np.nan
        else:
            row["r2_cv"] = np.nan
        rows.append(row)

    # full-rank OLS as reference
    fit_ols = reduced_rank_regression(Y, X, min(X.shape[1], Y.shape[1]))
    curve = pd.DataFrame(rows)
    curve.attrs["r2_full"] = rrr_score(Y, X, fit_ols)
    return curve


# ---------------------------------------------------------------------------
# Static latent variable model
# ---------------------------------------------------------------------------
@dataclass
class FactorAnalysisResult:
    model: FactorAnalysis
    latents: np.ndarray            # (n_obs, n_factors)
    loadings: np.ndarray           # (n_units, n_factors)
    explained: np.ndarray          # per-unit fraction of variance from factors
    total_explained: float


def factor_analysis(Y, n_factors=3, seed=42) -> FactorAnalysisResult:
    """Factor analysis of a ``(n_obs, n_units)`` population matrix.

    ``explained`` reports, per unit, the fraction of its variance captured by
    the shared factors (sum of squared loadings / variance), which quantifies
    how population-level each neuron's activity is.
    """
    Y = np.asarray(Y, float)
    scaler = StandardScaler()
    Yz = scaler.fit_transform(Y)
    fa = FactorAnalysis(n_components=min(n_factors, Yz.shape[1]), random_state=seed)
    latents = fa.fit_transform(Yz)
    loadings = fa.components_.T
    var = Yz.var(axis=0)
    explained = (loadings ** 2).sum(axis=1) / np.maximum(var, 1e-12)
    return FactorAnalysisResult(
        model=fa, latents=latents, loadings=loadings, explained=explained,
        total_explained=float(np.sum(loadings ** 2) / np.sum(var)),
    )


# ---------------------------------------------------------------------------
# Latent dynamics
# ---------------------------------------------------------------------------
def fit_var1(traj):
    """First-order VAR fitted to latent trajectories.

    Parameters
    ----------
    traj : ndarray (n_conditions, n_time, n_latents)

    Transitions are built *within* conditions only, so the discontinuity
    between the end of one condition's trajectory and the start of the next
    never enters the fit.

    Returns
    -------
    dict with ``A`` (n_latents x n_latents), ``eigenvalues``, ``r2`` (one-step
    pooled R²), ``bin_size_used`` flag and the design matrices.
    """
    traj = np.asarray(traj, float)
    n_cond, n_time, n_lat = traj.shape
    if n_time < 2:
        raise ValueError("trajectories need at least two time bins")

    X = traj[:, :-1, :].reshape(-1, n_lat)
    Y = traj[:, 1:, :].reshape(-1, n_lat)
    A_T, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = A_T.T
    pred = X @ A_T
    sst = np.sum((Y - Y.mean(axis=0)) ** 2)
    r2 = 1 - np.sum((Y - pred) ** 2) / sst if sst > 0 else np.nan
    return {"A": A, "eigenvalues": np.linalg.eigvals(A), "r2": float(r2),
            "X": X, "Y": Y, "n_obs": len(X)}


def eigenvalue_summary(eigenvalues, bin_size):
    """Turn VAR eigenvalues into decay times and oscillation frequencies.

    Returns a DataFrame with magnitude (per-bin damping), the equivalent decay
    time constant and, for complex eigenvalues, the oscillation frequency in Hz.
    """
    rows = []
    for i, lam in enumerate(np.atleast_1d(eigenvalues)):
        mag = float(abs(lam))
        decay = np.nan if mag <= 0 or mag >= 1 else -bin_size / np.log(mag)
        freq = (abs(float(np.angle(lam))) / (2 * np.pi * bin_size)
                if np.imag(lam) != 0 else np.nan)
        rows.append({"eigen_index": i, "real": float(np.real(lam)),
                     "imag": float(np.imag(lam)), "magnitude": mag,
                     "decay_time_s": float(decay), "freq_hz": freq})
    return pd.DataFrame(rows)


def dynamic_factor(Y, n_factors=2, factor_order=1):
    """Linear-Gaussian dynamic factor model (AR(``factor_order``) latents).

    Fitted by maximum likelihood with :mod:`statsmodels`. Returns ``None`` if
    the optimizer fails to converge, so callers can degrade gracefully.
    """
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

    Y = np.asarray(Y, float)
    if Y.shape[1] < 2 or len(Y) < 20:
        return None
    try:
        model = DynamicFactor(Y, k_factors=int(n_factors),
                              factor_order=int(factor_order))
        res = model.fit(disp=False, maxiter=200)
    except Exception:
        return None
    return res


def select_latent_dimension(Y, candidates=None, cfg: AnalysisConfig | None = None):
    """AIC/BIC search over the number of dynamic factors."""
    cfg = cfg or AnalysisConfig()
    candidates = candidates or cfg.latent_factors
    rows = []
    for k in candidates:
        res = dynamic_factor(Y, k)
        if res is None:
            rows.append({"n_factors": int(k), "loglik": np.nan, "aic": np.nan,
                         "bic": np.nan, "converged": False})
            continue
        rows.append({"n_factors": int(k), "loglik": float(res.llf),
                     "aic": float(res.aic), "bic": float(res.bic),
                     "converged": True})
    return pd.DataFrame(rows)


def latent_trial_activity(Y, n_factors=3, seed=42):
    """Factor-analysis latents of the trial x unit response matrix.

    Complements the PCA in :mod:`vision_pipeline.population`: factor analysis
    separates shared (latent) from private (noise) variance, whereas PCA
    attributes all variance to the components.
    """
    return factor_analysis(Y, n_factors=n_factors, seed=seed)