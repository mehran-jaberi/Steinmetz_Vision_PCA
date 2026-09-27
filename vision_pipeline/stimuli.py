"""Reconstruction of the visual stimuli from the raw trial/event data.

Two stimulus streams are reconstructed:

* **Task gratings** - one row per trial with onset/offset, left/right contrast,
  signed/absolute contrast and a condition label.
* **Sparse noise** - the mapping of flash times onto a grid of stimulus cells.

Everything here is deterministic and side-effect free; the reconstruction
functions take a :class:`~vision_pipeline.io.SessionData` (or its raw arrays)
and return tidy DataFrames.

Methodological notes
--------------------
* ``trials.intervals`` gives the trial start/stop; the grating ``offset`` is
  approximated by the go cue, which is when the stimulus is turned off in this
  task. The measured ``stim_duration`` is exported so the approximation is
  auditable rather than hidden.
* The raw sparse-noise timestamp array is **not** sorted and positions are
  paired with timestamps by row index, so times and positions are sorted
  *jointly*. Sorting them independently would scramble the stimulus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import condition_label


@dataclass
class SparseNoiseGrid:
    """Geometry of the sparse-noise stimulus grid."""

    xs: np.ndarray
    ys: np.ndarray
    n_x: int
    n_y: int

    @property
    def n_cells(self) -> int:
        return self.n_x * self.n_y

    def reshape_map(self, values):
        """Reshape a ``(n_cells,)`` vector into a ``(n_x, n_y)`` map.

        The flat index convention (row-major over ``y``) matches
        :func:`reconstruct_sparse_noise`.
        """
        return np.asarray(values).reshape(self.n_y, self.n_x).T


def reconstruct_grating_stimuli(session) -> pd.DataFrame:
    """Reconstruct the per-trial drifting-grating stimulus presentations.

    Returns a DataFrame with one row per trial (including non-included/blank
    trials, which downstream stages filter explicitly).
    """
    tr = session.trial_info
    n = session.n_trials
    df = pd.DataFrame(
        {
            "trial_idx": np.arange(n),
            "onset": tr["visualStim_times"],
            "offset": tr["goCue_times"],
            "contrast_left": tr["visualStim_contrastLeft"],
            "contrast_right": tr["visualStim_contrastRight"],
            "response_choice": tr["response_choice"],
            "feedback": tr["feedbackType"],
            "repNum": tr["repNum"],
            "included": tr["included"].astype(bool),
        }
    )
    df["signed_contrast"] = df["contrast_right"] - df["contrast_left"]
    df["abs_contrast"] = np.maximum(df["contrast_left"], df["contrast_right"])
    df["condition"] = [
        condition_label(cl, cr)
        for cl, cr in zip(df["contrast_left"], df["contrast_right"])
    ]
    df["stim_duration"] = df["offset"] - df["onset"]
    df["is_blank"] = df["abs_contrast"] == 0
    df["is_unilateral"] = (df["contrast_left"] == 0) | (df["contrast_right"] == 0)
    return df


def reconstruct_sparse_noise(session):
    """Reconstruct the sparse-noise flash sequence and the grid geometry.

    Returns
    -------
    flashes : DataFrame
        Columns ``flash_idx, time, x, y, cell`` (time-sorted).
    grid : SparseNoiseGrid
    """
    positions = np.asarray(session.sparseNoise["positions"])
    times = np.asarray(session.sparseNoise["times"]).ravel()
    if len(positions) != len(times):
        raise ValueError(
            f"sparse-noise positions ({len(positions)}) and times ({len(times)}) "
            "have different lengths; they must be paired by row"
        )

    order = np.argsort(times, kind="stable")
    times_sorted = times[order]
    positions_sorted = positions[order]

    xs = np.unique(positions_sorted[:, 0])
    ys = np.unique(positions_sorted[:, 1])
    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_i = {y: i for i, y in enumerate(ys)}
    cell = np.array(
        [y_to_i[y] * len(xs) + x_to_i[x] for x, y in positions_sorted], dtype=int
    )

    flashes = pd.DataFrame(
        {
            "flash_idx": np.arange(len(times_sorted)),
            "time": times_sorted,
            "x": positions_sorted[:, 0],
            "y": positions_sorted[:, 1],
            "cell": cell,
        }
    )
    grid = SparseNoiseGrid(xs=xs, ys=ys, n_x=len(xs), n_y=len(ys))
    return flashes, grid


def extract_grating_representations(grating_df, features=None) -> pd.DataFrame:
    """Per-trial stimulus feature matrix (design matrix) for the gratings.

    By default the five features used by the original pipeline are returned:
    left contrast, right contrast, their interaction, signed contrast and
    absolute contrast.
    """
    if features is None:
        features = ["contrast_left", "contrast_right"]
    X = pd.DataFrame(index=grating_df.index)
    for f in features:
        X[f] = grating_df[f].values
    if "contrast_left" in features and "contrast_right" in features:
        X["interaction"] = grating_df["contrast_left"].values * grating_df[
            "contrast_right"
        ].values
    if "signed_contrast" not in X.columns:
        X["signed_contrast"] = grating_df["signed_contrast"].values
    if "abs_contrast" not in X.columns:
        X["abs_contrast"] = grating_df["abs_contrast"].values
    return X


def build_stimulus_movie(flashes, grid, bin_size=0.1):
    """Time-binned ``(n_bins, n_cells)`` binary stimulus movie.

    A bin is ``True`` for the cell that flashed in it. Bins with no flash are
    all-zero (blank screen), which is what makes the movie usable as a design
    matrix for a stimulus encoding model.

    Returns
    -------
    bins : ndarray (n_bins + 1,) bin edges in seconds
    movie : ndarray of bool (n_bins, n_cells)
    """
    times = flashes["time"].values
    if len(times) == 0:
        return np.array([0.0]), np.zeros((0, grid.n_cells), dtype=bool)

    t0, t1 = times.min(), times.max()
    bins = np.arange(t0, t1 + bin_size, bin_size)
    n_bins = max(len(bins) - 1, 1)
    movie = np.zeros((n_bins, grid.n_cells), dtype=bool)
    bi = np.clip(
        np.searchsorted(bins, times, side="right") - 1, 0, n_bins - 1
    )
    movie[bi, flashes["cell"].values] = True
    return bins, movie


def sparse_noise_occupancy(flashes, grid) -> np.ndarray:
    """Number of flashes delivered to each grid cell."""
    return np.bincount(
        flashes["cell"].values.astype(int), minlength=grid.n_cells
    ).astype(float)


def grating_use_mask(grating_df, require_included=True, require_stimulus=True):
    """Standard trial mask used by every stimulus-response stage.

    Parameters
    ----------
    require_included : bool
        Restrict to trials the experimenters marked as included.
    require_stimulus : bool
        Drop blank trials (absolute contrast == 0).
    """
    mask = np.ones(len(grating_df), dtype=bool)
    if require_included:
        mask &= grating_df["included"].values
    if require_stimulus:
        mask &= (grating_df["abs_contrast"].values > 0)
    return mask
