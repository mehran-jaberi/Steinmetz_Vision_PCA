"""Small, dependency-light numeric helpers shared across the pipeline.

These functions are pure (no session state) which makes them easy to unit test
and reuse. Behaviour is kept identical to the original ``PART2.py`` helpers so
that refactored results can be validated against the published outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Spike counting / statistics
# ---------------------------------------------------------------------------
def count_spikes_in_windows(spike_times, align_times, window):
    """Number of spikes in ``window`` = (start, stop) relative to each align time.

    Parameters
    ----------
    spike_times : array-like
        Sorted spike times (seconds).
    align_times : array-like
        Reference times, one per trial/event.
    window : tuple (start, stop)
        Window offsets in seconds, relative to each align time.

    Notes
    -----
    Fully vectorised over ``align_times`` via ``np.searchsorted``. The window is
    half-open: ``[start, stop)`` for the left edge and inclusive right edge, so
    a spike exactly at ``align + stop`` is counted.
    """
    spike_times = np.asarray(spike_times, float)
    start = np.asarray(align_times, float) + window[0]
    stop = np.asarray(align_times, float) + window[1]
    i0 = np.searchsorted(spike_times, start, side="left")
    i1 = np.searchsorted(spike_times, stop, side="right")
    return (i1 - i0).astype(float)


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Returns
    -------
    mask : ndarray of bool
        ``True`` for hypotheses rejected at FDR ``alpha``. NaN p-values are
        treated as non-significant.
    """
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    finite = ~np.isnan(pvals)
    if not np.any(finite):
        return mask

    p = pvals[finite]
    order = np.argsort(p)
    ranked = p[order]
    thr = (np.arange(1, p.size + 1) / p.size) * alpha
    sig = ranked <= thr
    if np.any(sig):
        k = int(np.max(np.where(sig)[0]))
        idx = np.flatnonzero(finite)[order[: k + 1]]
        mask[idx] = True
    return mask


def zscore(x, axis=0, eps=1e-12):
    """Z-score along ``axis`` with a numerical floor on the standard deviation."""
    x = np.asarray(x, float)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def circular_shift(values, offset, period):
    """Shift ``values`` by ``offset`` inside a periodic interval of length ``period``.

    Used to build empirical null distributions: circularly shifting a spike
    train (or its trial-aligned events) destroys the alignment to the stimulus
    while preserving the neuron's temporal firing-rate structure.
    """
    values = np.asarray(values, float)
    return (values - values.min() + offset) % period + values.min()


# ---------------------------------------------------------------------------
# Label / formatting helpers
# ---------------------------------------------------------------------------
def condition_label(cl, cr):
    """Human-readable stimulus-condition label, e.g. ``'L0.25 R0'``."""
    return f"L{cl:g} R{cr:g}"


def parse_condition(label):
    """Inverse of :func:`condition_label` -> ``(contrast_left, contrast_right)``."""
    left, right = str(label).split()
    return float(left[1:]), float(right[1:])


def condition_sort_key(label):
    """Sort key ordering conditions by (left contrast, right contrast)."""
    return parse_condition(label)


# ---------------------------------------------------------------------------
# Plotting / IO helpers
# ---------------------------------------------------------------------------
def save_fig(fig, path, dpi=150, verbose=True):
    """Save a figure to ``path`` (creating parents) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    saved figure: {path}")
    return path


def write_table(df, path, verbose=True):
    """Write a DataFrame to CSV (creating parents) and report the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"    saved table : {path} ({len(df)} rows)")
    return path
