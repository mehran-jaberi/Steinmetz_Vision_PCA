"""Diagnose stimulus-response synchrony for one or more sessions (scratch).

For each session it builds the sparse-noise movie and the refined population
response exactly as the reconstruction pipeline does, then reports the
lag-resolved correlation between the population firing rate and the number of
flashes per bin. A decoder that assumes the wrong latency fails exactly like a
session with no signal, so this separates the two cases.
"""

from __future__ import annotations

import sys

import numpy as np

from vision_pipeline import AnalysisConfig, load_session_alf
from vision_pipeline.responses import population_time_series
from vision_pipeline.stimuli import build_stimulus_movie, reconstruct_sparse_noise
from vision_pipeline.units import identify_visual_neurons


def diagnose(path, max_lag=40, verbose=True):
    cfg = AnalysisConfig()
    session = load_session_alf(path, verbose=False)
    units = identify_visual_neurons(session, cfg, verbose=False)
    flashes, grid = reconstruct_sparse_noise(session)
    edges, movie = build_stimulus_movie(flashes, grid, bin_size=cfg.movie_bin)
    Y = population_time_series(session, units.refined_ids, edges)

    rate = Y.sum(axis=1).astype(float)
    nflash = movie.sum(axis=1).astype(float)
    times = np.unique(np.asarray(session.sparseNoise["times"]).ravel())

    print(f"--- {path}")
    print(f"  grid {grid.n_x} x {grid.n_y} = {grid.n_cells} cells | "
          f"{len(Y)} bins | {len(units.refined_ids)} refined units")
    print(f"  flash bins {float((nflash > 0).mean()):.3f} | "
          f"flashes/bin {nflash.mean():.2f} | frames {len(times)} | "
          f"median frame interval {np.median(np.diff(times)):.4f} s")
    print(f"  spikes/bin {rate.mean():.2f} | per unit "
          f"{rate.mean() / max(Y.shape[1], 1):.4f}")

    r = rate - rate.mean()
    f = nflash - nflash.mean()
    best = (0, -2.0)
    for lag in range(max_lag + 1):
        c = float(np.corrcoef(r[lag:], f[:len(f) - lag])[0, 1])
        if c > best[1]:
            best = (lag, c)
        if verbose and lag < 10:
            print(f"    lag {lag:2d}: r = {c:+.4f}")
    print(f"  best lag {best[0]} bins (r = {best[1]:+.4f})")
    return {"path": str(path), "best_lag": best[0], "best_r": best[1],
            "n_cells": grid.n_cells, "bins": len(Y)}


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        diagnose(arg)