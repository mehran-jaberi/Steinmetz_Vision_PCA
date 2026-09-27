"""Experimental algorithms: reconstruct the visual input from neural activity.

**Task 1 of the experimental branch** - recreate the stimulus from the
population response. The target is the sparse-noise stimulus movie
(``n_bins x n_cells`` binary flashes; 297 cells on the usual 9 x 33 grid); the
input is the refined visual population (``n_bins x n_units`` spike counts).
Seven decoders are fitted and compared on held-out contiguous blocks of time
(see :mod:`vision_pipeline.reconstruction`).

Usage
-----
    uv run python experimental.py --selftest          # synthetic validation
    uv run python experimental.py                     # example session
    uv run python experimental.py --methods ridge,softmax --folds 5
    uv run python experimental.py --session <path> --out experimental_outputs

The self-test is the honest check that the machinery works: it builds a
synthetic linear-Poisson encoder with a known ground truth, verifies that every
decoder recovers it far better than a time-shifted control, and compares each
decoder with an oracle matched filter that is given the true kernels.

Everything the runner writes goes to ``experimental_outputs/`` (tables, figures,
predictions and a provenance manifest).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from vision_pipeline import (AnalysisConfig, DEFAULT_SESSION,
                             load_session_alf)
from vision_pipeline.config import PROJECT_ROOT
from vision_pipeline.reconstruction import (METHODS, ReconstructionConfig,
                                            cell_centres,
                                            chance_control, contiguous_folds,
                                            cross_validate_reconstruction,
                                            evaluate_reconstruction,
                                            lagged_features)
from vision_pipeline.reporting import write_manifest
from vision_pipeline.responses import population_time_series
from vision_pipeline.stimuli import (SparseNoiseGrid, build_stimulus_movie,
                                     reconstruct_sparse_noise)
from vision_pipeline.units import identify_visual_neurons
from vision_pipeline.utils import write_table

DEFAULT_OUT = PROJECT_ROOT / "experimental_outputs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="experimental.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                       help="run the synthetic validation (no data needed)")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                        help="session folder to reconstruct from")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", default=",".join(METHODS),
                        help=f"comma-separated subset of {METHODS}")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--pre", type=int, default=None,
                        help="response bins before the stimulus bin")
    parser.add_argument("--post", type=int, default=None,
                        help="response bins after it (non-causal look-ahead)")
    parser.add_argument("--no-control", action="store_true",
                        help="skip the time-shifted chance control")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args) -> ReconstructionConfig:
    over = {}
    if args.folds:
        over["n_folds"] = int(args.folds)
    if args.pre is not None:
        over["pre"] = int(args.pre)
    if args.post is not None:
        over["post"] = int(args.post)
    return replace(ReconstructionConfig(), **over)


def best_method(summary: pd.DataFrame) -> str:
    """Name of the decoder with the highest pooled top-1 localisation accuracy."""
    acc = np.asarray(summary["top1_accuracy"], dtype=float)
    return str(summary["method"].iloc[int(np.argmax(acc))])


# ---------------------------------------------------------------------------
# Synthetic validation of the decoders
# ---------------------------------------------------------------------------
def synthetic_dataset(n_units=30, n_bins=2400, sigma=1.5, gain=6.0,
                      flash_rate=0.12, seed=1):
    """A sparse-noise experiment with a known linear-Poisson encoder.

    Each unit has a Gaussian spatial kernel around a random preferred cell and
    fires Poisson counts with rate ``exp(-3 + gain * K @ s(t - 1))`` - a
    stimulus-to-response latency of exactly one bin. Returns ``(Y, movie, grid,
    K)`` so the true kernels are available as the oracle.
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.arange(9.0), np.arange(33.0)
    grid = SparseNoiseGrid(xs=xs, ys=ys, n_x=len(xs), n_y=len(ys))
    n_cells = grid.n_cells

    movie = np.zeros((n_bins, n_cells), bool)
    flashed = rng.choice(n_bins, size=int(flash_rate * n_bins), replace=False)
    movie[flashed, rng.integers(0, n_cells, size=len(flashed))] = True

    x_c, y_c = cell_centres(grid)
    K = np.zeros((n_units, n_cells))
    for u in range(n_units):
        centre = int(rng.integers(0, n_cells))
        d2 = (x_c - x_c[centre]) ** 2 + (y_c - y_c[centre]) ** 2
        K[u] = np.exp(-d2 / (2.0 * sigma ** 2))

    lam = np.exp(-3.0 + gain * (movie[:-1].astype(float) @ K.T))
    Y = np.zeros((n_bins, n_units))
    Y[1:] = rng.poisson(lam)
    return Y, movie, grid, K


def oracle_scores(Y, K, cfg):
    """Matched filter using the *true* kernels: the ceiling for a learned decoder."""
    scores = np.zeros((len(Y), K.shape[1]))
    scores[:-cfg.lag] = np.asarray(Y, float)[cfg.lag:] @ np.asarray(K, float)
    return scores


def _fold_leakage(cfg, n_bins):
    """Any train/test pair sharing bins or sitting closer than the feature window."""
    problems = []
    for train, test in contiguous_folds(n_bins, cfg):
        if np.intersect1d(train, test).size:
            problems.append("train/test overlap")
        if train.size and test.size:
            closest = int(np.abs(train[:, None] - test[None, :]).min())
            if closest <= cfg.pre + cfg.post:
                problems.append(f"gap {closest} <= pre+post {cfg.pre + cfg.post}")
    return problems


def run_selftest(cfg=None, methods=METHODS, verbose=True) -> dict:
    """Validate every decoder against ground truth, oracle and chance control."""
    cfg = cfg or ReconstructionConfig()
    Y, movie, grid, K = synthetic_dataset()
    _, valid = lagged_features(Y, cfg)

    problems = _fold_leakage(cfg, len(Y))
    if problems:
        raise AssertionError(f"cross-validation leakage: {set(problems)}")

    oracle = evaluate_reconstruction(oracle_scores(Y, K, cfg), movie, grid,
                                     valid=valid)
    summary, folds, predictions = cross_validate_reconstruction(
        Y, movie, grid, cfg, methods=methods, verbose=False)
    control, _ = chance_control(Y, movie, grid, cfg, method="ridge")

    ctrl_auc = float(control["detection_auc"].iloc[0])
    ctrl_top1 = float(control["top1_accuracy"].iloc[0])
    chance_top1 = 1.0 / grid.n_cells

    rows, ok = [], True
    for method in methods:
        row = summary[summary["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        # a decoder is working if it beats its own time-shifted control clearly
        passed = bool(r["detection_auc"] > ctrl_auc + 0.05
                      and r["top1_accuracy"] > max(1.5 * ctrl_top1, chance_top1 * 1.5))
        ok &= passed
        rows.append({"method": method, "detection_auc": r["detection_auc"],
                     "top1_accuracy": r["top1_accuracy"],
                     "top5_accuracy": r["top5_accuracy"],
                     "control_auc": ctrl_auc, "control_top1": ctrl_top1,
                     "pass": passed})
    table = pd.DataFrame(rows)

    if verbose:
        print("  synthetic validation (30 units, 2400 bins, known kernels)")
        print(f"    oracle (true kernels) : AUC {oracle['detection_auc']:.3f}, "
              f"top-1 {oracle['top1_accuracy']:.2%}, "
              f"err {oracle['localisation_error']:.2f} cells")
        print(f"    chance control (ridge): AUC {ctrl_auc:.3f}, "
              f"top-1 {ctrl_top1:.2%} (chance {chance_top1:.2%})")
        for _, r in table.iterrows():
            print(f"    {r['method']:8s} AUC {r['detection_auc']:.3f} | "
                  f"top-1 {r['top1_accuracy']:.2%} | "
                  f"top-5 {r['top5_accuracy']:.2%} | "
                  f"{'PASS' if r['pass'] else 'FAIL'}")
        print(f"    no cross-validation leakage, "
              f"{len(methods)} decoders checked")
    return {"ok": ok, "table": table, "oracle": oracle, "control": control,
            "summary": summary, "folds": folds}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_reconstruction(summary, predictions, movie, grid, out_dir, control=None,
                        verbose=True):
    """Figure 13: quality per decoder. Figure 14: held-out reconstruction examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vision_pipeline.utils import save_fig

    frame = summary if control is None or control.empty else pd.concat(
        [summary, control], ignore_index=True)
    colors = ["#bdbdbd" if "control" in str(m) else "#3182bd"
              for m in frame["method"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    panels = [("detection_auc", "Detection AUC"),
              ("top1_accuracy", "Top-1 localisation accuracy"),
              ("mean_cell_correlation", "Mean per-cell correlation")]
    for ax, (col, label) in zip(axes, panels):
        ax.bar(frame["method"], frame[col].astype(float), color=colors)
        if col == "top1_accuracy":
            ax.axhline(1.0 / grid.n_cells, color="black", ls="--", lw=1,
                       label="chance")
            ax.legend(fontsize=8)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
    fig.suptitle("Reconstruction quality on held-out time blocks",
                 fontweight="bold")
    fig.tight_layout()
    saved = [save_fig(fig, out_dir / "fig13_reconstruction_quality.png",
                      verbose=verbose)]

    # --- examples for the best localiser ---
    best = best_method(summary)
    scores = predictions[best]["scores"]
    valid_rows = np.flatnonzero(np.isfinite(scores).all(axis=1))
    flash_rows = valid_rows[movie[valid_rows].any(axis=1)]
    if len(flash_rows) == 0:
        return saved

    start = max(int(flash_rows[len(flash_rows) // 2]) - 10, 0)
    win = np.arange(start, min(start + 60, len(movie)))
    win = win[np.isin(win, valid_rows)]
    pred_cells = scores[win].argmax(axis=1)
    true_cells, true_bins = [], []
    for i, t in enumerate(win):
        for c in np.flatnonzero(movie[t]):
            true_cells.append(int(c))
            true_bins.append(i)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].scatter(true_cells, true_bins, s=8, color="#08519c", label="true flash")
    axes[0].scatter(pred_cells, np.arange(len(win)), s=16, marker="x",
                    color="#e6550d", label="decoded (argmax)")
    axes[0].set_xlabel("stimulus cell")
    axes[0].set_ylabel("held-out bin")
    axes[0].set_title(f"True vs decoded flash ({best})")
    axes[0].legend(fontsize=8)

    example = int(np.median(flash_rows))
    axes[1].imshow(grid.reshape_map(movie[example].astype(float)),
                   origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("true flash map (one bin)")
    axes[2].imshow(grid.reshape_map(scores[example]), origin="lower",
                   aspect="auto", cmap="viridis")
    axes[2].set_title(f"decoded map ({best})")
    for ax in axes[1:]:
        ax.set_xlabel("x cell")
        ax.set_ylabel("y cell")
    fig.suptitle("Held-out reconstruction examples", fontweight="bold")
    fig.tight_layout()
    saved.append(save_fig(fig, out_dir / "fig14_reconstruction_examples.png",
                          verbose=verbose))
    return saved


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------
def run_session(session_path, out_dir, cfg=None, rcfg=None, methods=METHODS,
                control=True, figures=True, verbose=True) -> pd.DataFrame:
    """Reconstruct the sparse-noise movie from one session's visual population."""
    cfg = cfg or AnalysisConfig()
    rcfg = rcfg or ReconstructionConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXPERIMENTAL TASK 1: RECONSTRUCT THE STIMULUS FROM NEURAL RESPONSE")
    print("=" * 78)
    session = load_session_alf(session_path, verbose=verbose)
    units = identify_visual_neurons(session, cfg, verbose=verbose)
    refined = units.refined_ids
    if len(refined) < cfg.min_units:
        raise SystemExit(f"only {len(refined)} refined units (<{cfg.min_units}); "
                         "nothing to decode")

    flashes, grid = reconstruct_sparse_noise(session)
    edges, movie = build_stimulus_movie(flashes, grid, bin_size=cfg.movie_bin)
    Y = population_time_series(session, refined, edges)
    if verbose:
        print(f"\n  Target : stimulus movie {movie.shape[0]} bins x "
              f"{movie.shape[1]} cells ({cfg.movie_bin:g} s bins, "
              f"{int(movie.sum())} flashes)")
        print(f"  Input  : population response {Y.shape[0]} bins x "
              f"{Y.shape[1]} refined units ({int(Y.sum())} spikes)")
        print(f"  Decoder: response window {rcfg.pre} bins before to "
              f"{rcfg.post} after the stimulus bin (latency {rcfg.lag} bins), "
              f"{rcfg.n_folds}-fold contiguous CV, gap {rcfg.gap}\n")

    t0 = time.perf_counter()
    summary, folds, predictions = cross_validate_reconstruction(
        Y, movie, grid, rcfg, methods=methods, verbose=verbose)
    elapsed = time.perf_counter() - t0

    control_summary = pd.DataFrame()
    if control:
        if verbose:
            print("  chance control (time-shifted target):")
        control_summary, _ = chance_control(Y, movie, grid, rcfg, method="ridge",
                                            verbose=verbose)

    write_table(summary, out_dir / "reconstruction_summary.csv", verbose=verbose)
    write_table(folds, out_dir / "reconstruction_folds.csv", verbose=verbose)
    if not control_summary.empty:
        write_table(control_summary, out_dir / "reconstruction_control.csv",
                    verbose=verbose)

    best = best_method(summary)
    top1 = float(summary.loc[summary["method"] == best, "top1_accuracy"].iloc[0])
    np.savez_compressed(out_dir / "reconstruction_predictions.npz",
                        method=str(best),
                        scores=predictions[best]["scores"].astype(np.float32),
                        detection=predictions[best]["detection"].astype(np.float32),
                        movie=movie.astype(np.uint8))
    if verbose:
        print(f"  best localiser: {best} (top-1 {top1:.2%})")

    if figures:
        plot_reconstruction(summary, predictions, movie, grid, out_dir,
                            control=control_summary, verbose=verbose)

    write_manifest(out_dir / "manifest.json", "reconstruction", rcfg, extra={
        "session": str(session_path),
        "n_units": int(len(refined)),
        "n_bins": int(movie.shape[0]),
        "n_cells": int(movie.shape[1]),
        "n_flashes": int(movie.sum()),
        "bin_size_s": float(cfg.movie_bin),
        "methods": list(methods),
        "best_method": str(best),
        "seconds": float(elapsed),
    })
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    rcfg = config_from_args(args)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; choose from {METHODS}")

    if args.selftest:
        result = run_selftest(rcfg, methods=methods, verbose=not args.quiet)
        return 0 if result["ok"] else 1

    run_session(args.session, args.out, rcfg=rcfg, methods=methods,
                control=not args.no_control, figures=not args.no_figures,
                verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())