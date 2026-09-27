"""Response matrices: trial x unit spike counts, PSTHs and binned tensors."""

from __future__ import annotations

import numpy as np

from .utils import count_spikes_in_windows


def build_response_matrix(session, unit_ids, align_times, window):
    """``(n_align, n_units)`` matrix of spike counts in ``window`` per event.

    Parameters
    ----------
    session : SessionData
    unit_ids : sequence of int
    align_times : array-like
        One reference time (s) per row of the output matrix.
    window : tuple (start, stop)
        Offsets (s) relative to each align time.
    """
    unit_ids = np.asarray(unit_ids, dtype=int)
    align_times = np.asarray(align_times, float)
    X = np.zeros((len(align_times), len(unit_ids)), dtype=float)
    for j, uid in enumerate(unit_ids):
        X[:, j] = count_spikes_in_windows(
            session.unit_spike_times(int(uid)), align_times, window
        )
    return X


def binned_response_tensor(session, unit_ids, align_times, window, bin_size):
    """``(n_align, n_bins, n_units)`` tensor of spike counts in consecutive bins.

    Used for time-resolved population analyses. The bin edges are shared across
    trials, so ``bins[k]`` is the same latency window for every trial.
    """
    unit_ids = np.asarray(unit_ids, dtype=int)
    align_times = np.asarray(align_times, float)
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    n_bins = len(edges) - 1
    tensor = np.zeros((len(align_times), n_bins, len(unit_ids)), dtype=float)
    for j, uid in enumerate(unit_ids):
        s = session.unit_spike_times(int(uid))
        for b in range(n_bins):
            tensor[:, b, j] = count_spikes_in_windows(
                s, align_times, (edges[b], edges[b + 1])
            )
    return edges, tensor


def bin_edges(window, bin_size):
    """Shared bin edges for a window, mirroring :func:`binned_response_tensor`."""
    return np.arange(window[0], window[1] + bin_size, bin_size)


def bin_spike_counts(spike_times, edges):
    """Histogram spike times into ``edges`` (half-open bins, O(n_spikes)).

    Much cheaper than calling :func:`count_spikes_in_windows` once per bin,
    which matters for the model-based analyses on whole-session time series.
    """
    spike_times = np.asarray(spike_times, float)
    edges = np.asarray(edges, float)
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return np.zeros(0)
    idx = np.searchsorted(edges, spike_times, side="right") - 1
    valid = (idx >= 0) & (idx < n_bins)
    return np.bincount(idx[valid], minlength=n_bins).astype(float)


def population_time_series(session, unit_ids, edges):
    """``(n_bins, n_units)`` spike-count time series over shared bin edges."""
    unit_ids = np.asarray(unit_ids, dtype=int)
    n_bins = len(edges) - 1
    Y = np.zeros((n_bins, len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        Y[:, j] = bin_spike_counts(session.unit_spike_times(int(uid)), edges)
    return Y


def peri_stimulus_psth(session, unit_id, align_times, window, bin_size,
                       trial_mask=None):
    """Mean firing rate (spikes/s) of one unit aligned to ``align_times``.

    Returns
    -------
    centers : ndarray
    rate : ndarray (spikes/s)
    sem : ndarray (spikes/s)
    counts : ndarray (n_trials, n_bins) raw counts
    """
    align_times = np.asarray(align_times, float)
    if trial_mask is not None:
        align_times = align_times[np.asarray(trial_mask, bool)]

    edges = bin_edges(window, bin_size)
    n_bins = len(edges) - 1
    s = session.unit_spike_times(int(unit_id))
    counts = np.zeros((len(align_times), n_bins))
    for b in range(n_bins):
        counts[:, b] = count_spikes_in_windows(
            s, align_times, (edges[b], edges[b + 1])
        )

    rate = counts.mean(axis=0) / bin_size
    sem = counts.std(axis=0, ddof=1) / np.sqrt(max(len(align_times), 1)) / bin_size
    centers = edges[:-1] + bin_size / 2
    return centers, rate, sem, counts


def condition_average(tensor, conditions, min_trials=3):
    """Condition-average a ``(n_trials, n_bins, n_units)`` tensor.

    Returns
    -------
    labels : list of str   condition labels (order preserved)
    means  : ndarray (n_labels, n_bins, n_units)
    n      : ndarray (n_labels,) trial counts
    """
    conditions = np.asarray(conditions)
    labels = [c for c in dict.fromkeys(conditions.tolist())
              if int((conditions == c).sum()) >= min_trials]
    means = np.zeros((len(labels), tensor.shape[1], tensor.shape[2]))
    n = np.zeros(len(labels), dtype=int)
    for i, c in enumerate(labels):
        m = conditions == c
        means[i] = tensor[m].mean(axis=0)
        n[i] = int(m.sum())
    return labels, means, n