"""Receptive fields via spike-triggered averaging on the sparse-noise grid.

The spike-triggered average (STA) at grid cell ``c`` is the mean number of
spikes fired in the ``win`` seconds following a flash at ``c``, normalised by
that cell's occupancy. Two significance tests are computed:

* ``significant`` - the original pipeline's test: a Gaussian Monte-Carlo
  threshold on ``max|z|`` across cells. It assumes the 297 cell values are
  independent standard normals, which they are not (overlapping response
  windows and correlated noise inflate the null).
* ``significant_perm`` - an empirical null built by **circularly shifting** the
  unit's spike train, which preserves its firing-rate structure and its
  autocorrelation while destroying the flash alignment. This is the test the
  pipeline reports as its primary result; the Gaussian test is kept so results
  remain comparable with earlier runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .utils import bh_fdr


@dataclass
class ReceptiveFieldResult:
    table: pd.DataFrame                 # one row per unit
    maps: dict                          # unit -> {"sta":, "z":}
    grid: object
    z_threshold_gaussian: float
    n_permutations: int


def compute_receptive_fields(session, unit_ids, flashes, grid,
                             cfg: AnalysisConfig | None = None,
                             verbose=True) -> ReceptiveFieldResult:
    """STA receptive fields plus Gaussian and permutation significance."""
    cfg = cfg or AnalysisConfig()
    unit_ids = [int(u) for u in unit_ids]

    times = flashes["time"].values
    cells = flashes["cell"].values.astype(int)
    n_cells = grid.n_cells
    occupancy = np.bincount(cells, minlength=n_cells).astype(float)

    win = cfg.rf_window
    t_lo = times.min() - win
    t_hi = times.max() + win
    period = t_hi - t_lo

    # --- Gaussian Monte-Carlo threshold (legacy test) ---
    rng = np.random.default_rng(cfg.random_seed)
    null_max = np.max(np.abs(rng.standard_normal((cfg.rf_null_samples, n_cells))),
                      axis=1)
    z_thresh = float(np.percentile(null_max, 95))

    rows = []
    maps = {}
    for uid in unit_ids:
        spikes = session.unit_spike_times(uid)
        spikes = spikes[(spikes >= t_lo) & (spikes <= t_hi)]
        if len(spikes) == 0:
            rows.append({"cluster_id": uid, "z_max": np.nan, "significant": False,
                         "z_max_perm": np.nan, "p_perm": np.nan,
                         "significant_perm": False, "peak_x": np.nan,
                         "peak_y": np.nan, "n_flash_spikes": 0})
            continue

        sta, z = _sta_map(spikes, times, cells, occupancy, win, n_cells)
        z_max = float(np.max(np.abs(z)))
        peak = int(np.argmax(z))

        p_perm = np.nan
        z_max_perm = np.nan
        if cfg.rf_permutations > 0:
            null = _shift_null(spikes, times, cells, occupancy, win, n_cells,
                               period, cfg.rf_permutations,
                               np.random.default_rng(cfg.random_seed + uid))
            z_max_perm = float(np.max(null))
            p_perm = (1 + int(np.sum(null >= z_max))) / (cfg.rf_permutations + 1)

        rows.append({
            "cluster_id": uid,
            "z_max": z_max,
            "significant": bool(z_max > z_thresh),
            "z_max_perm": z_max_perm,
            "p_perm": p_perm,
            "significant_perm": bool(p_perm < cfg.fdr_alpha) if np.isfinite(p_perm)
            else False,
            "peak_x": grid.xs[peak % grid.n_x],
            "peak_y": grid.ys[peak // grid.n_x],
            "n_flash_spikes": int(np.sum(
                _counts_in_window(spikes, times, win))),
        })
        maps[uid] = {"sta": sta, "z": z}

    table = pd.DataFrame(rows)
    # per-unit permutation p-values are FDR-corrected across the population
    # (the permutation null is already per-unit empirical, so no Gaussian
    # assumption is involved)
    table["significant_perm"] = bh_fdr(
        table["p_perm"].fillna(1.0).values, alpha=cfg.fdr_alpha
    )
    # z_max of the permutation null, reported for transparency
    if verbose:
        n_g = int(table["significant"].sum())
        n_p = int(table["significant_perm"].sum())
        print(f"  Receptive fields: {n_p}/{len(table)} units significant "
              f"(permutation FDR p<{cfg.fdr_alpha}); "
              f"{n_g}/{len(table)} by the Gaussian max|z| > {z_thresh:.2f} test")
    return ReceptiveFieldResult(table=table, maps=maps, grid=grid,
                                z_threshold_gaussian=z_thresh,
                                n_permutations=cfg.rf_permutations)


def _counts_in_window(spikes, times, win):
    """Spike counts in ``[t, t + win)`` for each flash time ``t``."""
    i0 = np.searchsorted(spikes, times, side="left")
    i1 = np.searchsorted(spikes, times + win, side="right")
    return (i1 - i0).astype(float)


def _sta_map(spikes, times, cells, occupancy, win, n_cells):
    """STA and its within-unit z-score over the grid.

    ``spikes`` must be sorted; a cheap endpoint check catches the "rotated but
    unsorted" mistake that silently corrupts ``np.searchsorted``.
    """
    spikes = np.asarray(spikes, float)
    if len(spikes) > 1 and spikes[-1] < spikes[0]:
        raise ValueError("_sta_map requires sorted spike times")
    counts = _counts_in_window(spikes, times, win)
    summed = np.bincount(cells, weights=counts, minlength=n_cells)
    sta = summed / np.maximum(occupancy, 1)
    z = (sta - sta.mean()) / (sta.std() + 1e-12)
    return sta, z


def _shift_null(spikes, times, cells, occupancy, win, n_cells, period,
                n_perm, rng):
    """Empirical max|z| null from circularly shifted spike trains.

    The shifted train is re-sorted before counting: a circular shift is a
    rotation, not a monotone transform, and ``np.searchsorted`` requires sorted
    input. Skipping the sort silently produces garbage STAs (this actually
    inflated the null by ~3x before being caught by the calibration check in
    :mod:`vision_pipeline.validation`).
    """
    null = np.empty(n_perm)
    base = spikes.min()
    for i in range(n_perm):
        offset = rng.uniform(0, period)
        shifted = np.sort((spikes - base + offset) % period + base)
        _, z = _sta_map(shifted, times, cells, occupancy, win, n_cells)
        null[i] = np.max(np.abs(z))
    return null


def centroid_of_map(z, grid):
    """Centre of mass of the positive part of an STA map, in grid coordinates.

    Useful as a summary of where a unit's receptive field sits; returns NaN for
    units with no positive deflection.
    """
    w = np.clip(np.asarray(z, float), 0, None)
    total = w.sum()
    if total <= 0:
        return np.nan, np.nan
    x_centers = np.repeat(grid.xs, grid.n_y)
    y_centers = np.tile(grid.ys, grid.n_x)
    return float((w * x_centers).sum() / total), float((w * y_centers).sum() / total)


def population_rf_image(result: ReceptiveFieldResult, significant_only=True):
    """Mean z-scored STA map across units (a crude population RF)."""
    units = result.table
    ids = units.loc[units["significant_perm"] if significant_only
                    else units.index, "cluster_id"].tolist()
    maps = [result.maps[int(u)]["z"] for u in ids if int(u) in result.maps]
    if not maps:
        return None
    return np.mean(maps, axis=0)


def example_units(result: ReceptiveFieldResult, n=6, only_significant=True):
    """Cluster ids to show in the RF example figure."""
    t = result.table
    if only_significant and t["significant_perm"].any():
        t = t[t["significant_perm"]]
    t = t.dropna(subset=["z_max"]).sort_values("z_max", ascending=False)
    return t["cluster_id"].astype(int).head(n).tolist()