"""Model-based population analysis 1/2: Poisson stimulus encoding models (GLM).

A per-neuron Poisson generalized linear model maps the sparse-noise stimulus
movie onto the unit's spike-count time series::

    log E[y(t)] = b0 + sum_c k_c * s_c(t - d) + sum_h a_h * y(t - h)

where ``s_c`` is the binary indicator for grid cell ``c`` (lagged by ``d`` bins
to allow for response latency), ``k`` is the unit's linear receptive-field
kernel, and the ``a`` terms are a short autoregressive spike history. This is a
linear-nonlinear (Poisson) encoding model: fitting it per neuron and measuring
*held-out* deviance explained is a quantitative complement to the descriptive
STA analysis in :mod:`vision_pipeline.receptive_fields`.

Notes
-----
* The stimulus movie, the spike counts and the design matrix all use the same
  shared bin edges (passed in explicitly), which is the property that makes the
  regression a valid encoding model. :mod:`vision_pipeline.validation` checks
  this alignment.
* The model is fitted with an L2 penalty (``cfg.glm_alpha``); the
  likelihood-ratio test and the cross-validated deviance both account for the
  extra parameters, which the in-sample R² of the earlier pipeline did not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold

from .config import AnalysisConfig
from .responses import bin_spike_counts
from .utils import bh_fdr


@dataclass
class GLMResult:
    table: pd.DataFrame            # one row per unit
    filters: dict                  # unit -> stimulus kernel (n_cells,)
    history: dict                  # unit -> history coefficients
    feature_names: list
    edges: np.ndarray              # shared bin edges used by the design
    n_bins: int


def build_stimulus_design(movie, edges, counts, cfg: AnalysisConfig | None = None,
                          lag_bins: int = 1):
    """Regression design for the sparse-noise GLM.

    Parameters
    ----------
    movie : ndarray of bool, shape (n_bins, n_cells)
        Stimulus movie from :func:`vision_pipeline.stimuli.build_stimulus_movie`.
    edges : ndarray, shape (n_bins + 1,)
        Bin edges used to build ``movie`` and to bin the spike counts.
    counts : ndarray, shape (n_bins,)
        Spike counts of the modelled unit on the same bins.
    lag_bins : int
        Stimulus lag (in bins); rows are trimmed so that every row of the
        returned design refers to a spike bin ``t`` with predictors from
        ``t - lag_bins``.

    Returns
    -------
    design : ndarray (n_bins - lag_bins, n_features + history columns)
    target : ndarray (n_bins - lag_bins,) spike counts aligned to ``design``
    names : list of feature names
    """
    cfg = cfg or AnalysisConfig()
    movie = np.asarray(movie, float)
    counts = np.asarray(counts, float)
    n_bins = movie.shape[0]
    if len(edges) != n_bins + 1:
        raise ValueError("edges must have len(n_bins) + 1 entries")
    if len(counts) != n_bins:
        raise ValueError("counts must be aligned to the movie bins")

    lag = int(max(lag_bins, 0))
    n_rows = n_bins - lag
    if n_rows <= 0:
        raise ValueError("movie is shorter than the requested stimulus lag")

    target = counts[lag:]
    blocks = [movie[:n_rows]]                       # stimulus at t - lag
    names = [f"cell{c}" for c in range(movie.shape[1])]

    for h in range(1, cfg.glm_history_bins + 1):
        hist = np.zeros(n_rows)
        # row i is spike bin i + lag (matching `target`); history uses i + lag - h
        src = np.arange(n_rows) + lag - h
        ok = src >= 0
        hist[ok] = counts[src[ok]]
        blocks.append(hist.reshape(-1, 1))
        names.append(f"history{h}")

    return np.column_stack(blocks), target, names


def _poisson_loglik(y, mu):
    """Poisson log-likelihood ignoring the constant ``-log(y!)`` term."""
    mu = np.clip(mu, 1e-9, None)
    return float(np.sum(y * np.log(mu) - mu))


def _deviance(y, mu):
    """Poisson deviance of ``mu`` relative to the saturated model."""
    mu = np.clip(mu, 1e-9, None)
    y = np.asarray(y, float)
    terms = np.zeros_like(y)
    nz = y > 0
    terms[nz] = y[nz] * np.log(y[nz] / mu[nz])
    return float(2 * np.sum(terms - (y - mu)))


def fit_neuron_glm(y, design, cfg: AnalysisConfig | None = None,
                   n_splits: int = 5):
    """Fit one Poisson GLM and return fit statistics.

    Returns a dict with:

    ``r2_dev``     in-sample deviance explained relative to an intercept-only model
    ``r2_dev_cv``  the same quantity on held-out folds (honest fit)
    ``lr_stat`` / ``lr_p``  likelihood-ratio test against the intercept-only model
    ``coef``       coefficients (stimulus kernel then history terms)
    ``intercept``  baseline log-rate
    """
    cfg = cfg or AnalysisConfig()
    y = np.asarray(y, float)
    n = len(y)
    folds = int(min(n_splits, max(n // 10, 2)))

    empty = {
        "fit": False, "r2_dev": np.nan, "r2_dev_cv": np.nan, "loglik": np.nan,
        "lr_stat": np.nan, "lr_p": np.nan, "n_spikes": float(y.sum()),
        "coef": None, "intercept": np.nan, "n_bins": n,
    }
    if n < 20 or folds < 2 or y.sum() < 5:
        return empty

    try:
        model = PoissonRegressor(alpha=cfg.glm_alpha, max_iter=500, tol=1e-6)
        model.fit(design, y)
    except Exception:
        return empty

    mu = model.predict(design)
    loglik = _poisson_loglik(y, mu)
    mu0 = np.full(n, max(y.mean(), 1e-9))
    loglik0 = _poisson_loglik(y, mu0)
    dev, dev0 = _deviance(y, mu), _deviance(y, mu0)
    r2_dev = 1 - dev / dev0 if dev0 > 0 else np.nan

    lr_stat = max(2 * (loglik - loglik0), 0.0)
    lr_p = float(stats.chi2.sf(lr_stat, df=design.shape[1]))

    kf = KFold(n_splits=folds, shuffle=True, random_state=cfg.random_seed)
    dev_test, dev0_test = 0.0, 0.0
    for train, test in kf.split(design):
        try:
            m = PoissonRegressor(alpha=cfg.glm_alpha, max_iter=500, tol=1e-6)
            m.fit(design[train], y[train])
            mu_t = m.predict(design[test])
        except Exception:
            return empty
        dev_test += _deviance(y[test], mu_t)
        dev0_test += _deviance(y[test], np.full(len(test), max(y[train].mean(), 1e-9)))
    r2_dev_cv = 1 - dev_test / dev0_test if dev0_test > 0 else np.nan

    return {
        "fit": True, "r2_dev": float(r2_dev), "r2_dev_cv": float(r2_dev_cv),
        "loglik": loglik, "lr_stat": float(lr_stat), "lr_p": lr_p,
        "n_spikes": float(y.sum()), "coef": model.coef_,
        "intercept": float(model.intercept_), "n_bins": n,
    }


def fit_population_glms(session, unit_ids, movie, edges, grid,
                        cfg: AnalysisConfig | None = None, lag_bins: int = 1,
                        verbose=True) -> GLMResult:
    """Fit the sparse-noise Poisson GLM to every unit in ``unit_ids``."""
    cfg = cfg or AnalysisConfig()
    movie = np.asarray(movie, bool)
    n_bins = movie.shape[0]
    n_cells = grid.n_cells

    rows, filters, history = [], {}, {}
    names: list = []

    for uid in unit_ids:
        uid = int(uid)
        counts = bin_spike_counts(session.unit_spike_times(uid), edges)[:n_bins]
        design, target, names = build_stimulus_design(
            movie, edges, counts, cfg, lag_bins=lag_bins
        )
        res = fit_neuron_glm(target, design, cfg)

        row = {"cluster_id": uid, "n_spikes": res["n_spikes"],
               "n_bins": res["n_bins"]}
        for key in ("r2_dev", "r2_dev_cv", "lr_stat", "lr_p", "loglik", "fit"):
            row[key] = res.get(key, np.nan)
        rows.append(row)

        if res["coef"] is not None:
            filters[uid] = np.asarray(res["coef"][:n_cells], float)
            history[uid] = np.asarray(res["coef"][n_cells:], float)

    table = pd.DataFrame(rows)
    table["significant"] = bh_fdr(table["lr_p"].fillna(1.0).values,
                                  alpha=cfg.fdr_alpha)

    if verbose:
        ok = table["fit"].fillna(False).astype(bool)
        if ok.any():
            print(f"  Poisson GLM: fitted {int(ok.sum())}/{len(table)} units; "
                  f"{int(table['significant'].sum())} significantly "
                  f"stimulus-driven (FDR p<{cfg.fdr_alpha}); median held-out "
                  f"deviance explained = {table.loc[ok, 'r2_dev_cv'].median():.3f}")
        else:
            print("  Poisson GLM: no unit could be fitted")

    return GLMResult(table=table, filters=filters, history=history,
                     feature_names=names, edges=np.asarray(edges, float),
                     n_bins=n_bins)


def glm_vs_sta_agreement(glm_result: GLMResult, rf_result) -> pd.DataFrame:
    """Correlate GLM stimulus kernels with the STA maps of the same units.

    The two receptive-field estimates come from different procedures
    (regularized model fit on spike counts vs. spike-triggered average), so
    their agreement is an internal consistency check on the RF analysis.
    """
    rows = []
    for uid, kernel in glm_result.filters.items():
        if int(uid) not in rf_result.maps:
            continue
        sta = np.asarray(rf_result.maps[int(uid)]["sta"], float)
        if np.std(kernel) == 0 or np.std(sta) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(kernel, sta)[0, 1])
        rows.append({"cluster_id": int(uid), "kernel_sta_corr": corr,
                     "kernel_norm": float(np.linalg.norm(kernel)),
                     "sta_norm": float(np.linalg.norm(sta))})
    return pd.DataFrame(rows)