"""
PART2.py -- Part 2: Vision-specific population analysis (Steinmetz et al., 2019)

This script implements the next stage of the analysis pipeline, moving from
general session exploration (see Main_Steinmetz.ipynb) to a quantitative,
population-level treatment of the visual component of the data.

Pipeline
--------
1. Refine visual-neuron identification
   * Anatomical: every unit is mapped to a brain region through its peak
     recording channel (channels.brainLocation.tsv / Allen ontology) and
     restricted to visual areas (VIS*).
   * Quality: only well-isolated units are kept
     (clusters._phy_annotation == 1  ->  good units).
   * Functional: units must be significantly driven by the task-grating
     stimulus onset (paired t-test across stimulus trials, FDR-corrected).

2. Reconstruct visual stimuli
   * Task gratings: per-trial onset/offset and left/right contrast.
   * Sparse noise: the 9 x 33 position grid plus flash onset times
     (the raw timestamp array is unsorted and is sorted here).

3. Extract quantitative visual stimulus representations
   * Per-trial stimulus design matrices (left contrast, right contrast,
     interaction, signed contrast, absolute contrast, condition labels).
   * Sparse-noise flash sequence, time-binned stimulus movie and per-cell
     occupancy.

4. Relate stimulus representations to neural activity
   * Trial-by-trial response matrices (spike counts in a stimulus window).
   * Contrast tuning curves and a tuning "surface" over (left, right) contrast.
   * Per-neuron linear regression of responses on stimulus features.
   * Decoding of stimulus side from population activity.
   * Receptive fields via spike-triggered averages on the sparse-noise grid.

5. PCA / dimensionality reduction
   * PCA of the trial x neuron population response.
   * Condition structure in PC space (scree, PC scatter, within-vs-across
     condition distance ratio).
   * Time-resolved population trajectories (PCA over time x neurons).

All figures and result tables are written to ./part2_outputs
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="ticks", context="notebook")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent
DATA_ROOT = WORKSPACE / "Steinmetz_et_al_2019_9974357" / "nicklab" / "Subjects"
SESSION = DATA_ROOT / "Cori" / "2016-12-14" / "001"
OUTPUT_DIR = WORKSPACE / "part2_outputs"

# Stimulus-response windows (seconds relative to stimulus onset)
STIM_WINDOW = (0.05, 0.35)      # s after visual-stimulus onset
BASE_WINDOW = (-0.35, -0.05)    # s before stimulus onset

# Unit-selection criteria
GOOD_ANNOTATION = 1             # phy annotation: 1 = good, 2 = MUA, 3 = noise
VISUAL_AREAS = {"VIS", "VISa", "VISal", "VISam", "VISl", "VISli", "VISmma",
                "VISmmp", "VISp", "VISpl", "VISpm", "VISpor", "VISrl"}
MIN_TOTAL_SPIKES = 50           # sanity filter on total spike count
FDR_ALPHA = 0.05
RF_WINDOW = 0.1                 # spike-triggered-average integration (s)
RF_NULL_SAMPLES = 1000          # Monte-Carlo samples for RF significance


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def count_spikes_in_windows(spike_times, align_times, window):
    """Number of spikes in ``window`` (start, stop) relative to each align time.

    ``spike_times`` must be sorted. Fully vectorised over ``align_times``.
    """
    start = np.asarray(align_times, float) + window[0]
    stop = np.asarray(align_times, float) + window[1]
    i0 = np.searchsorted(spike_times, start, side="left")
    i1 = np.searchsorted(spike_times, stop, side="right")
    return (i1 - i0).astype(float)


def bh_fdr(pvals, alpha=FDR_ALPHA):
    """Benjamini-Hochberg FDR correction. Returns a boolean significant mask."""
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thr = (np.arange(1, n + 1) / n) * alpha
    sig = ranked <= thr
    mask = np.zeros(n, dtype=bool)
    if np.any(sig):
        k = int(np.max(np.where(sig)[0]))
        mask[order[: k + 1]] = True
    return mask


def condition_label(cl, cr):
    """Human-readable stimulus-condition label, e.g. 'L0.25 R0'."""
    return f"L{cl:g} R{cr:g}"


def save_fig(fig, name):
    """Save a figure to the output directory and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved figure: part2_outputs/{name}")


# ----------------------------------------------------------------------------
# 0. DATA LOADING
# ----------------------------------------------------------------------------
def load_session_alf(session_folder):
    """Load an ALF-format session into a single dict.

    Returns
    -------
    data : dict with keys
        spike_times, spike_clusters, n_clusters,
        cluster_depths, cluster_peak_channel, cluster_probes,
        cluster_annotation, cluster_waveform_duration,
        channel_positions, channel_probe, channel_region,
        probe_names, probe_insertion,
        trial_info (dict of arrays), n_trials,
        passiveVisual (dict), sparseNoise (dict), session
    """
    folder = Path(session_folder) / "alf"
    if not folder.exists():
        raise FileNotFoundError(f"No 'alf' folder found at {folder}")

    data = {}
    data["session"] = str(Path(session_folder))

    # --- spikes ---
    data["spike_times"] = np.load(folder / "spikes.times.npy").ravel()
    data["spike_clusters"] = np.load(folder / "spikes.clusters.npy").ravel().astype(int)

    # --- cluster metadata ---
    data["n_clusters"] = len(np.load(folder / "clusters.depths.npy"))
    data["cluster_depths"] = np.load(folder / "clusters.depths.npy").ravel()
    data["cluster_peak_channel"] = np.load(folder / "clusters.peakChannel.npy").ravel().astype(int)
    data["cluster_probes"] = np.load(folder / "clusters.probes.npy").ravel().astype(int)
    data["cluster_waveform_duration"] = np.load(folder / "clusters.waveformDuration.npy").ravel()
    data["cluster_annotation"] = np.load(folder / "clusters._phy_annotation.npy",
                                          allow_pickle=True).ravel().astype(int)

    # --- channel metadata + brain regions ---
    data["channel_positions"] = np.load(folder / "channels.sitePositions.npy")
    data["channel_probe"] = np.load(folder / "channels.probe.npy").ravel().astype(int)
    bl = pd.read_csv(folder / "channels.brainLocation.tsv", sep="\t")
    data["channel_region"] = bl["allen_ontology"].astype(str).values
    data["channel_ccf"] = bl[["ccf_ap", "ccf_dv", "ccf_lr"]].values

    # --- probe names (e.g. 'V1', 'M2') ---
    pf = pd.read_csv(folder / "probes.rawFilename.tsv", sep="\t")
    data["probe_names"] = [str(n) for n in pf["rawFilename"].values]
    data["probe_insertion"] = pd.read_csv(folder / "probes.insertion.tsv", sep="\t")

    # --- trial data ---
    trial_files = {
        "goCue_times": "goCue_times", "visualStim_times": "visualStim_times",
        "response_times": "response_times", "feedback_times": "feedback_times",
        "visualStim_contrastLeft": "visualStim_contrastLeft",
        "visualStim_contrastRight": "visualStim_contrastRight",
        "response_choice": "response_choice", "feedbackType": "feedbackType",
        "repNum": "repNum", "included": "included",
    }
    trial_info = {}
    for key, fname in trial_files.items():
        trial_info[key] = np.load(folder / f"trials.{fname}.npy").ravel()
    trial_info["intervals"] = np.load(folder / "trials.intervals.npy")
    data["trial_info"] = trial_info
    data["n_trials"] = len(trial_info["goCue_times"])

    # --- passive visual stimuli ---
    data["passiveVisual"] = {
        "times": np.load(folder / "passiveVisual.times.npy").ravel(),
        "contrastLeft": np.load(folder / "passiveVisual.contrastLeft.npy").ravel(),
        "contrastRight": np.load(folder / "passiveVisual.contrastRight.npy").ravel(),
    }

    # --- sparse noise stimuli (positions are paired with times by row) ---
    data["sparseNoise"] = {
        "positions": np.load(folder / "sparseNoise.positions.npy"),
        "times": np.load(folder / "sparseNoise.times.npy").ravel(),
    }

    # sanity checks
    assert data["spike_clusters"].max() < data["n_clusters"], \
        "spike cluster id exceeds number of clusters"
    assert data["cluster_peak_channel"].max() < len(data["channel_region"]), \
        "peak channel id exceeds number of channels"

    print(f"Loaded session: {data['session']}")
    print(f"  {data['n_trials']} trials | {data['n_clusters']} clusters | "
          f"{len(data['spike_times']):,} spikes")
    return data


def unit_spike_times(data, unit_id):
    """Cached per-unit spike times."""
    cache = data.setdefault("_spike_cache", {})
    if unit_id not in cache:
        cache[unit_id] = data["spike_times"][data["spike_clusters"] == unit_id]
    return cache[unit_id]


def build_response_matrix(data, unit_ids, align_times, window=STIM_WINDOW):
    """(n_align, n_units) matrix of spike counts in ``window`` per trial."""
    align_times = np.asarray(align_times, float)
    X = np.zeros((len(align_times), len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        X[:, j] = count_spikes_in_windows(unit_spike_times(data, uid), align_times, window)
    return X


# ----------------------------------------------------------------------------
# 1. REFINED VISUAL-NEURON IDENTIFICATION
# ----------------------------------------------------------------------------
def identify_visual_neurons(data, alpha=FDR_ALPHA,
                            window=STIM_WINDOW, base_window=BASE_WINDOW):
    """Refined visual-neuron identification.

    Anatomical (region via peak channel) -> quality (phy annotation) ->
    functional (significant response to task-grating stimulus onset).

    Note: the passive-visual block was inspected for this session and did not
    reliably drive visual units, so the functional stage uses the task-grating
    responses instead (the same stimulus used for the downstream analysis).

    Returns
    -------
    visual_units : DataFrame, one row per refined visual unit
    visual_mask  : bool array over all clusters
    summary      : dict with refinement-stage counts
    """
    region = data["channel_region"][data["cluster_peak_channel"]]
    probe_names = data["probe_names"]

    n = data["n_clusters"]
    visual_mask = np.zeros(n, dtype=bool)

    # per-cluster total spike counts
    n_spikes_per_cluster = np.bincount(data["spike_clusters"], minlength=n)

    rows = []
    for uid in range(n):
        rows.append({
            "cluster_id": uid,
            "probe": int(data["cluster_probes"][uid]),
            "probe_name": probe_names[int(data["cluster_probes"][uid])],
            "region": region[uid],
            "depth": data["cluster_depths"][uid],
            "peak_channel": int(data["cluster_peak_channel"][uid]),
            "quality": int(data["cluster_annotation"][uid]),
            "n_spikes": int(n_spikes_per_cluster[uid]),
        })
    units = pd.DataFrame(rows)

    # ---- Stage A: anatomical (visual-area region) ----
    units["in_visual_area"] = units["region"].isin(VISUAL_AREAS)
    n_anatomical = int(units["in_visual_area"].sum())

    # ---- Stage B: + quality (good units only) ----
    units["is_good"] = units["quality"] == GOOD_ANNOTATION
    units["passes_basic"] = units["in_visual_area"] & units["is_good"] & \
        (units["n_spikes"] >= MIN_TOTAL_SPIKES)
    n_quality = int(units["passes_basic"].sum())

    # ---- Stage C: + functional (grating-driven responsiveness) ----
    tr = data["trial_info"]
    stim_times = tr["visualStim_times"][
        (tr["visualStim_contrastLeft"] > 0) | (tr["visualStim_contrastRight"] > 0)]
    print(f"\n  Stimulus trials used for responsiveness: {len(stim_times)}")

    candidates = units.index[units["passes_basic"]].tolist()
    resp_rates = np.zeros(len(candidates))
    base_rates = np.zeros(len(candidates))
    pvals = np.full(len(candidates), np.nan)
    tvals = np.full(len(candidates), np.nan)
    mods = np.zeros(len(candidates))

    for k, uid in enumerate(candidates):
        s = unit_spike_times(data, uid)
        r = count_spikes_in_windows(s, stim_times, window) / (window[1] - window[0])
        b = count_spikes_in_windows(s, stim_times, base_window) / \
            (base_window[1] - base_window[0])
        resp_rates[k] = r.mean()
        base_rates[k] = b.mean()
        mods[k] = (r.mean() - b.mean()) / (r.mean() + b.mean() + 1e-6)
        try:
            t, p = stats.ttest_rel(r, b)
            tvals[k], pvals[k] = t, p
        except Exception:
            tvals[k], pvals[k] = np.nan, np.nan

    # FDR correction across the candidate set
    pvals_ok = ~np.isnan(pvals)
    sig = np.zeros(len(candidates), dtype=bool)
    sig[pvals_ok] = bh_fdr(pvals[pvals_ok], alpha=alpha)

    units.loc[candidates, "visual_resp_rate"] = resp_rates
    units.loc[candidates, "visual_base_rate"] = base_rates
    units.loc[candidates, "visual_t"] = tvals
    units.loc[candidates, "visual_p"] = pvals
    units.loc[candidates, "visual_modulation"] = mods
    units.loc[candidates, "visually_responsive"] = sig

    # refined set = anatomical & good & responsive
    refined = (units["in_visual_area"] & units["is_good"] &
               (units["n_spikes"] >= MIN_TOTAL_SPIKES) &
               units["visually_responsive"].fillna(False).astype(bool))
    units["visual_refined"] = refined
    n_functional = int(refined.sum())
    refined_ids = units.loc[refined, "cluster_id"].values
    visual_mask[refined_ids] = True

    summary = {
        "n_anatomical": n_anatomical,
        "n_quality": n_quality,
        "n_functional": n_functional,
        "n_total": n,
    }

    # ---- Figures ----
    # (a) refinement cascade
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    stage_names = ["Anatomical\n(VIS areas)", "+ Quality\n(good units)",
                   "+ Responsive\n(grating onset)"]
    stage_counts = [n_anatomical, n_quality, n_functional]
    axes[0].bar(stage_names, stage_counts, color=["#6baed6", "#3182bd", "#08519c"],
                edgecolor="white")
    for i, c in enumerate(stage_counts):
        axes[0].text(i, c + 0.5, str(c), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of units")
    axes[0].set_title("Refinement cascade", fontweight="bold")

    # (b) region composition of the refined set
    comp = units.loc[units["visual_refined"], "region"].value_counts()
    axes[1].bar(range(len(comp)), comp.values, color="#74c476", edgecolor="white")
    axes[1].set_xticks(range(len(comp)))
    axes[1].set_xticklabels(comp.index, rotation=45, ha="right")
    axes[1].set_ylabel("Number of units")
    axes[1].set_title("Refined visual set: region composition", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig01_visual_neuron_refinement.png")

    # (c) depth profile of visual units with responsiveness overlay
    vis = units[units["in_visual_area"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(vis["depth"], vis["visual_modulation"], s=14, alpha=0.55,
               c=np.where(vis["visually_responsive"].fillna(False), "#e6550d", "#9ecae1"))
    ax.set_xlabel("Depth along probe (µm)")
    ax.set_ylabel("Grating modulation index")
    ax.set_title("Visual-area units: responsiveness vs depth", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", ls="", color="#e6550d", label="responsive"),
        plt.Line2D([0], [0], marker="o", ls="", color="#9ecae1", label="not responsive"),
    ], loc="best")
    fig.tight_layout()
    save_fig(fig, "fig01b_responsiveness_vs_depth.png")

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    units.to_csv(OUTPUT_DIR / "visual_units.csv", index=False)

    print("\n[Step 1] Refined visual-neuron identification")
    print(f"  {n_anatomical} units in visual areas "
          f"-> {n_quality} good-quality -> {n_functional} visually responsive (FDR p<{alpha})")
    return units, visual_mask, summary


# ----------------------------------------------------------------------------
# 2. STIMULUS RECONSTRUCTION
# ----------------------------------------------------------------------------
def reconstruct_grating_stimuli(data):
    """Reconstruct the per-trial drifting-grating stimulus presentations."""
    tr = data["trial_info"]
    n = data["n_trials"]
    df = pd.DataFrame({
        "trial_idx": np.arange(n),
        "onset": tr["visualStim_times"],
        "offset": tr["goCue_times"],
        "contrast_left": tr["visualStim_contrastLeft"],
        "contrast_right": tr["visualStim_contrastRight"],
        "response_choice": tr["response_choice"],
        "feedback": tr["feedbackType"],
        "included": tr["included"].astype(bool),
    })
    df["signed_contrast"] = df["contrast_right"] - df["contrast_left"]
    df["abs_contrast"] = np.maximum(df["contrast_left"], df["contrast_right"])
    df["condition"] = [condition_label(cl, cr) for cl, cr in
                       zip(df["contrast_left"], df["contrast_right"])]
    df["stim_duration"] = df["offset"] - df["onset"]
    return df


def reconstruct_sparse_noise(data):
    """Reconstruct the sparse-noise flash sequence.

    The raw timestamp array is NOT sorted; positions are paired with times by
    row index, so the two are sorted together here.
    """
    pos = data["sparseNoise"]["positions"]
    times = data["sparseNoise"]["times"].ravel()
    order = np.argsort(times)
    times, pos = times[order], pos[order]

    xs = np.unique(pos[:, 0])
    ys = np.unique(pos[:, 1])
    x_to_i = {x: i for i, x in enumerate(xs)}
    y_to_i = {y: i for i, y in enumerate(ys)}
    cell = np.array([y_to_i[y] * len(xs) + x_to_i[x] for x, y in pos])

    flashes = pd.DataFrame({
        "flash_idx": np.arange(len(times)),
        "time": times,
        "x": pos[:, 0],
        "y": pos[:, 1],
        "cell": cell,
    })
    grid = {"xs": xs, "ys": ys, "n_x": len(xs), "n_y": len(ys),
            "n_cells": len(xs) * len(ys)}
    return flashes, grid


# ----------------------------------------------------------------------------
# 3. QUANTITATIVE STIMULUS REPRESENTATIONS
# ----------------------------------------------------------------------------
def extract_grating_representations(grating_df):
    """Per-trial stimulus feature matrix (design matrix) for the gratings."""
    X = pd.DataFrame(index=grating_df.index)
    X["contrast_left"] = grating_df["contrast_left"].values
    X["contrast_right"] = grating_df["contrast_right"].values
    X["interaction"] = X["contrast_left"] * X["contrast_right"]
    X["signed_contrast"] = grating_df["signed_contrast"].values
    X["abs_contrast"] = grating_df["abs_contrast"].values
    return X


def build_stimulus_movie(flashes, grid, bin_size=0.1):
    """Time-binned (n_bins, n_cells) binary stimulus movie."""
    t0, t1 = flashes["time"].min(), flashes["time"].max()
    bins = np.arange(t0, t1 + bin_size, bin_size)
    n_bins = len(bins) - 1
    movie = np.zeros((n_bins, grid["n_cells"]), dtype=bool)
    bi = np.clip(np.searchsorted(bins, flashes["time"].values, side="right") - 1,
                 0, n_bins - 1)
    movie[bi, flashes["cell"].values] = True
    return bins, movie


# ----------------------------------------------------------------------------
# 4. RELATING STIMULUS REPRESENTATIONS TO NEURAL ACTIVITY
# ----------------------------------------------------------------------------
def compute_contrast_tuning(data, unit_ids, grating_df, window=STIM_WINDOW):
    """Mean response (spikes/window) per (left, right) contrast condition.

    Returns
    -------
    tuning     : DataFrame (condition, n_trials)
    matrix     : (n_conditions, n_units) mean response per condition
    sem_matrix : (n_conditions, n_units) SEM across trials
    uniq       : condition labels
    X, conds   : trial-level response matrix and condition labels (for reuse)
    """
    use = grating_df["included"] & (grating_df["abs_contrast"] > 0)
    align = grating_df.loc[use, "onset"].values
    X = build_response_matrix(data, unit_ids, align, window)

    conds = grating_df.loc[use, "condition"].values
    uniq = pd.unique(conds)

    rows = []
    matrix = np.full((len(uniq), len(unit_ids)), np.nan)
    sem_matrix = np.full((len(uniq), len(unit_ids)), np.nan)
    for i, cond in enumerate(uniq):
        m = conds == cond
        matrix[i] = X[m].mean(axis=0)
        sem_matrix[i] = X[m].std(axis=0) / np.sqrt(m.sum())
        rows.append({"condition": cond, "n_trials": int(m.sum())})
    tuning = pd.DataFrame(rows)
    return tuning, matrix, sem_matrix, uniq, X, conds


REG_FEATURES = ["contrast_left", "contrast_right", "interaction"]


def regress_responses_on_stimulus(data, unit_ids, grating_df, design, window=STIM_WINDOW):
    """Per-neuron OLS: response ~ left + right + interaction (+ intercept)."""
    use = grating_df["included"] & (grating_df["abs_contrast"] > 0)
    align = grating_df.loc[use, "onset"].values
    X = build_response_matrix(data, unit_ids, align, window)

    feats = design.loc[use, REG_FEATURES].values        # (n, k) no intercept
    A = np.column_stack([np.ones(len(feats)), feats])   # (n, k+1)
    k = feats.shape[1]
    n = len(feats)
    dof = n - k - 1

    cols = ["intercept"] + list(REG_FEATURES)
    rows = []
    for j, uid in enumerate(unit_ids):
        y = X[:, j]
        y_mean = y.mean()
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot == 0:
            rows.append({"cluster_id": uid, "r2": np.nan, "p": np.nan,
                         "side_preference": np.nan})
            continue
        betas, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ betas
        ss_res = np.sum((y - yhat) ** 2)
        r2 = 1 - ss_res / ss_tot
        F = (r2 / k) / ((1 - r2) / dof) if r2 < 1 else np.inf
        p = 1 - stats.f.cdf(F, k, dof) if np.isfinite(F) else 0.0
        row = {"cluster_id": uid, "r2": r2, "p": p,
               "side_preference": betas[2] - betas[1]}   # bR - bL
        for cname, b in zip(cols, betas):
            row[f"beta_{cname}"] = b
        rows.append(row)
    reg = pd.DataFrame(rows)
    reg["significant"] = bh_fdr(reg["p"].fillna(1.0).values, alpha=FDR_ALPHA)
    return reg, X


def decode_stimulus_side(X, grating_df, use, cv_folds=5):
    """Cross-validated logistic regression: right vs left stimulus.

    ``X`` rows must be aligned to the ``use`` mask of ``grating_df``.
    Only unilateral trials (one side stimulated) are decoded.
    """
    cl = grating_df.loc[use, "contrast_left"].values
    cr = grating_df.loc[use, "contrast_right"].values
    unilateral = (cl == 0) | (cr == 0)
    y = (cr[unilateral] > 0).astype(int)
    Xs = X[unilateral]
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std(), len(y)


def compute_receptive_fields(data, unit_ids, flashes, grid,
                             win=RF_WINDOW, n_null=RF_NULL_SAMPLES):
    """Spike-triggered-average receptive fields on the sparse-noise grid.

    For each unit, the STA at grid cell c is the mean number of spikes fired
    in the ``win`` seconds after a flash at cell c, normalised by occupancy.
    Significance is assessed against the 95th percentile of the null
    distribution of max |z| over the grid (Monte Carlo).
    """
    times = flashes["time"].values
    cells = flashes["cell"].values
    n_cells = grid["n_cells"]
    occ = np.bincount(cells, minlength=n_cells).astype(float)

    t_lo = times.min() - win
    t_hi = times.max() + win

    rng = np.random.default_rng(42)
    null_max = np.max(np.abs(rng.standard_normal((n_null, n_cells))), axis=1)
    thresh = float(np.percentile(null_max, 95))

    rows = []
    sta_maps = {}
    for uid in unit_ids:
        s = unit_spike_times(data, uid)
        s = s[(s >= t_lo) & (s <= t_hi)]
        if len(s) == 0:
            rows.append({"cluster_id": uid, "z_max": np.nan, "significant": False,
                         "peak_x": np.nan, "peak_y": np.nan, "n_flash_spikes": 0})
            continue
        i0 = np.searchsorted(s, times, side="left")
        i1 = np.searchsorted(s, times + win, side="right")
        counts = (i1 - i0).astype(float)
        sum_counts = np.bincount(cells, weights=counts, minlength=n_cells)
        sta = sum_counts / np.maximum(occ, 1)
        sta_mean = sta.mean()
        sta_std = sta.std()
        z = (sta - sta_mean) / (sta_std + 1e-12)
        z_max = np.max(np.abs(z))
        sig = z_max > thresh
        peak = int(np.argmax(z))
        rows.append({
            "cluster_id": uid, "z_max": z_max, "significant": sig,
            "peak_x": grid["xs"][peak % grid["n_x"]],
            "peak_y": grid["ys"][peak // grid["n_x"]],
            "n_flash_spikes": int(counts.sum()),
        })
        sta_maps[uid] = {"sta": sta, "z": z}

    rf = pd.DataFrame(rows)
    return rf, sta_maps, thresh


# ----------------------------------------------------------------------------
# 5. PCA / DIMENSIONALITY REDUCTION
# ----------------------------------------------------------------------------
def population_pca(X, conditions, n_components=10):
    """PCA on the trial x unit response matrix, with condition structure."""
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, Xz.shape[1]))
    scores = pca.fit_transform(Xz)
    evr = pca.explained_variance_ratio_

    # within vs across condition distance ratio in PC1-3
    s3 = scores[:, :3]
    conds = np.asarray(conditions)
    uniq = [c for c in pd.unique(conds)]
    within, across, n_pairs = [], [], 0
    for c in uniq:
        m = conds == c
        if m.sum() < 2:
            continue
        d = np.linalg.norm(s3[m][:, None, :] - s3[m][None, :, :], axis=2)
        within.append(d[np.triu_indices(m.sum(), 1)].mean())
    for i, c1 in enumerate(uniq):
        m1 = conds == c1
        for c2 in uniq[i + 1:]:
            m2 = conds == c2
            if m1.sum() and m2.sum():
                d = np.linalg.norm(s3[m1][:, None, :] - s3[m2][None, :, :], axis=2)
                across.append(d.mean())
    ratio = (np.mean(within) / np.mean(across)) if within and across else np.nan

    results = {
        "pca": pca, "scaler": scaler, "scores": scores, "evr": evr,
        "within_across_ratio": ratio,
        "discriminability": 1 - ratio if np.isfinite(ratio) else np.nan,
        "cum_evr": np.cumsum(evr),
        "within_mean": np.mean(within) if within else np.nan,
        "across_mean": np.mean(across) if across else np.nan,
    }
    return results


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("PART 2 - VISION-SPECIFIC POPULATION ANALYSIS")
    print("=" * 78)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- load ----------------
    data = load_session_alf(SESSION)

    # ---------------- Step 1 ----------------
    print("\n" + "#" * 78)
    print("# STEP 1: REFINED VISUAL-NEURON IDENTIFICATION")
    print("#" * 78)
    units_df, visual_mask, step1_summary = identify_visual_neurons(data)
    refined_ids = units_df.loc[units_df["visual_refined"], "cluster_id"].values
    print(f"  Refined visual population: {len(refined_ids)} units")

    if len(refined_ids) < 5:
        print("\n  WARNING: refined visual population is too small (<5 units); "
              "aborting downstream analysis. Check the session/quality filters.")
        return

    # ---------------- Step 2 ----------------
    print("\n" + "#" * 78)
    print("# STEP 2: VISUAL STIMULUS RECONSTRUCTION")
    print("#" * 78)
    grating_df = reconstruct_grating_stimuli(data)
    flashes, grid = reconstruct_sparse_noise(data)
    grating_df.to_csv(OUTPUT_DIR / "grating_stimuli.csv", index=False)
    flashes.to_csv(OUTPUT_DIR / "flashes.csv", index=False)
    print(f"  Grating trials        : {len(grating_df)} "
          f"(included: {grating_df['included'].sum()})")
    print(f"  Stimulus conditions   : {grating_df['condition'].nunique()}")
    print(f"  Sparse-noise flashes  : {len(flashes)} on a "
          f"{grid['n_x']} x {grid['n_y']} grid")

    # ---------------- Step 3 ----------------
    print("\n" + "#" * 78)
    print("# STEP 3: QUANTITATIVE STIMULUS REPRESENTATIONS")
    print("#" * 78)
    design = extract_grating_representations(grating_df)
    design.to_csv(OUTPUT_DIR / "stimulus_design.csv", index=False)
    movie_bins, movie = build_stimulus_movie(flashes, grid, bin_size=0.1)
    occ = np.bincount(flashes["cell"].values, minlength=grid["n_cells"])
    print(f"  Grating design matrix : {design.shape[1]} features x {design.shape[0]} trials")
    print(f"  Stimulus movie        : {movie.shape[0]} time bins x {movie.shape[1]} cells")
    print(f"  Sparse-noise occupancy: {occ.min()}-{occ.max()} flashes per cell")

    # ---- Step 3 figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(design.loc[grating_df["included"]].values.T,
                         aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(design.shape[1]))
    axes[0].set_yticklabels(design.columns)
    axes[0].set_xlabel("Trial (included)")
    axes[0].set_title("Grating design matrix", fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(occ.reshape(grid["n_y"], grid["n_x"]).T, origin="lower",
                   aspect="auto", cmap="magma")
    axes[1].set_xlabel("Y position (grid)")
    axes[1].set_ylabel("X position (grid)")
    axes[1].set_title("Sparse-noise occupancy", fontweight="bold")

    sub = flashes[(flashes["time"] < flashes["time"].min() + 30)]
    axes[2].scatter(sub["time"], sub["y"], s=1, color="black", alpha=0.3)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y position")
    axes[2].set_title("First 30 s of sparse-noise flashes", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig03_stimulus_representations.png")

    # ---------------- Step 4 ----------------
    print("\n" + "#" * 78)
    print("# STEP 4: RELATING STIMULUS REPRESENTATIONS TO NEURAL ACTIVITY")
    print("#" * 78)

    # (a) contrast tuning
    tuning, tun_matrix, _, uniq_conds, X_resp, conds = \
        compute_contrast_tuning(data, refined_ids, grating_df)

    # save a tidy (unit x condition) table
    tidy = pd.DataFrame(tun_matrix.T, columns=uniq_conds)
    tidy.insert(0, "cluster_id", refined_ids)
    tidy.to_csv(OUTPUT_DIR / "tuning.csv", index=False)
    print(f"  Contrast tuning: {len(uniq_conds)} conditions x {len(refined_ids)} units")

    # ---- tuning figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = sorted(range(len(uniq_conds)),
                   key=lambda i: (float(uniq_conds[i].split()[0][1:]),
                                  float(uniq_conds[i].split()[1][1:])))
    conds_order = [uniq_conds[i] for i in order]
    means = tun_matrix[order]
    pop_mean = means.mean(axis=1)
    pop_sem = means.std(axis=1) / np.sqrt(len(refined_ids))
    axes[0].errorbar(range(len(conds_order)), pop_mean, yerr=pop_sem,
                     fmt="o-", color="#08519c", capsize=3)
    axes[0].set_xticks(range(len(conds_order)))
    axes[0].set_xticklabels(conds_order, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Population mean response (spikes/0.3s)")
    axes[0].set_title("Population contrast tuning", fontweight="bold")

    # tuning surface: mean response across units vs (left, right) contrast
    cl_vals = np.sort(grating_df["contrast_left"].unique())
    cr_vals = np.sort(grating_df["contrast_right"].unique())
    surface = np.full((len(cl_vals), len(cr_vals)), np.nan)
    for i, c in enumerate(conds_order):
        cl = float(c.split()[0][1:])
        cr = float(c.split()[1][1:])
        surface[np.where(cl_vals == cl)[0][0], np.where(cr_vals == cr)[0][0]] = \
            means[i].mean()
    im1 = axes[1].imshow(surface, origin="lower", aspect="auto", cmap="YlOrRd")
    axes[1].set_xticks(range(len(cr_vals)))
    axes[1].set_xticklabels(cr_vals)
    axes[1].set_yticks(range(len(cl_vals)))
    axes[1].set_yticklabels(cl_vals)
    axes[1].set_xlabel("Right contrast")
    axes[1].set_ylabel("Left contrast")
    axes[1].set_title("Population tuning surface (mean spk/0.3s)", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    save_fig(fig, "fig04_contrast_tuning.png")

    # (b) regression of responses on stimulus features
    reg, X_reg = regress_responses_on_stimulus(data, refined_ids, grating_df, design)
    reg.to_csv(OUTPUT_DIR / "regression.csv", index=False)
    n_sig = int(reg["significant"].sum())
    print(f"  Regression: {n_sig}/{len(reg)} units significantly encode "
          f"stimulus contrast (FDR p<{FDR_ALPHA}); "
          f"median R2 = {reg['r2'].median():.3f}")

    # ---- regression figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(reg["r2"].dropna(), bins=25, color="#3182bd", edgecolor="white")
    axes[0].axvline(reg["r2"].median(), color="black", ls="--",
                    label=f"median = {reg['r2'].median():.3f}")
    axes[0].set_xlabel("R² (stimulus regression)")
    axes[0].set_ylabel("Number of units")
    axes[0].set_title("Stimulus encoding strength", fontweight="bold")
    axes[0].legend()

    good = units_df.set_index("cluster_id").loc[reg["cluster_id"]]
    axes[1].scatter(good["depth"].values, reg["side_preference"], s=12, alpha=0.5,
                    c=np.where(reg["significant"], "#e6550d", "#9ecae1"))
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Depth (µm)")
    axes[1].set_ylabel("Side preference  β(R) − β(L)")
    axes[1].set_title("Contrast-side preference vs depth", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig04b_stimulus_regression.png")

    # (c) decoding of stimulus side
    use_for_reg = grating_df["included"] & (grating_df["abs_contrast"] > 0)
    acc_mean, acc_std, n_side = decode_stimulus_side(X_reg, grating_df, use_for_reg)
    print(f"  Side decoding (unilateral trials, n={n_side}): "
          f"{acc_mean * 100:.1f} ± {acc_std * 100:.1f} %")

    # (d) sparse-noise receptive fields
    rf, sta_maps, rf_thresh = compute_receptive_fields(
        data, refined_ids, flashes, grid)
    rf.to_csv(OUTPUT_DIR / "receptive_fields.csv", index=False)
    n_rf = int(rf["significant"].sum())
    print(f"  Receptive fields: {n_rf}/{len(rf)} refined visual units have "
          f"significant RFs (max|z| > {rf_thresh:.2f})")

    # ---- RF figure ----
    sig_rf = rf[rf["significant"]]
    examples = sig_rf.head(6)["cluster_id"].values
    if len(examples) < 6 and len(sig_rf) > 0:
        examples = sig_rf["cluster_id"].values[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, uid in zip(axes, examples):
        zmap = sta_maps[uid]["z"].reshape(grid["n_y"], grid["n_x"]).T
        im = ax.imshow(zmap, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-np.nanmax(np.abs(zmap)), vmax=np.nanmax(np.abs(zmap)))
        ax.set_title(f"unit {uid}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(examples):]:
        ax.axis("off")
    fig.suptitle("Example sparse-noise receptive fields (z-scored STA)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig04c_receptive_fields.png")

    # ---- RF fraction figure ----
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Refined visual units"], [100 * n_rf / len(rf)], color="#74c476",
           edgecolor="white")
    ax.axhline(5, color="black", ls="--", lw=0.8)
    ax.set_ylabel("% with significant RF")
    ax.set_ylim(0, 105)
    ax.set_title(f"RF significance (threshold max|z| = {rf_thresh:.1f})",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig04d_rf_fraction.png")

    # ---------------- Step 5 ----------------
    print("\n" + "#" * 78)
    print("# STEP 5: PCA / DIMENSIONALITY REDUCTION")
    print("#" * 78)

    use = grating_df["included"] & (grating_df["abs_contrast"] > 0)
    align5 = grating_df.loc[use, "onset"].values
    X5 = build_response_matrix(data, refined_ids, align5, STIM_WINDOW)
    conds5 = grating_df.loc[use, "condition"].values
    print(f"  PCA on {X5.shape[0]} trials x {X5.shape[1]} units")

    pca_res = population_pca(X5, conds5)
    evr = pca_res["evr"]
    print(f"  Explained variance: PC1 = {evr[0] * 100:.1f}%, "
          f"PC1-2 = {pca_res['cum_evr'][1] * 100:.1f}%, "
          f"PC1-5 = {evr[:5].sum() * 100:.1f}%")
    print(f"  Condition discriminability (1 - within/across distance) "
          f"= {pca_res['discriminability']:.3f}")

    # ---- PCA figure ----
    scores = pca_res["scores"]
    uniq = list(pd.unique(conds5))
    palette = sns.color_palette("husl", len(uniq))
    color_map = dict(zip(uniq, palette))

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 1])

    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(1, len(evr) + 1), evr * 100, "o-", color="#08519c")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Scree plot", fontweight="bold")

    ax = fig.add_subplot(gs[1])
    for c in uniq:
        m = conds5 == c
        ax.scatter(scores[m, 0], scores[m, 1], s=22, alpha=0.75,
                   color=color_map[c], label=c, edgecolors="white", linewidths=0.3)
        if m.sum():
            ax.scatter(scores[m, 0].mean(), scores[m, 1].mean(), marker="X",
                       s=160, color="black", edgecolors="white", linewidths=0.8)
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    ax.set_title("Population responses in PC space", fontweight="bold")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))

    ax = fig.add_subplot(gs[2])
    ax.bar(["within", "across"], [pca_res["within_mean"], pca_res["across_mean"]],
           color=["#6baed6", "#fb6a4a"], edgecolor="white")
    ax.set_ylabel("Mean distance (PC1-3)")
    ax.set_title(f"Discriminability = {pca_res['discriminability']:.2f}",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig05_pca.png")

    # ---- time-resolved trajectories ----
    print("  Computing time-resolved population trajectories ...")
    tr_win = (-0.1, 0.5)
    tr_bin = 0.02
    tr_bins = np.arange(tr_win[0], tr_win[1] + tr_bin, tr_bin)
    n_tbins = len(tr_bins) - 1
    n_trials = int(use.sum())
    tensor = np.zeros((n_trials, n_tbins, len(refined_ids)))
    for u, uid in enumerate(refined_ids):
        s = unit_spike_times(data, uid)
        for b in range(n_tbins):
            tensor[:, b, u] = count_spikes_in_windows(
                s, align5, (tr_bins[b], tr_bins[b] + tr_bin))

    uniq_conds = [c for c in uniq if (conds5 == c).sum() >= 3]
    traj = np.zeros((len(uniq_conds), n_tbins, len(refined_ids)))
    for i, c in enumerate(uniq_conds):
        traj[i] = tensor[conds5 == c].mean(axis=0)

    Xt = traj.reshape(-1, len(refined_ids))
    Xt = StandardScaler().fit_transform(Xt)
    pca_t = PCA(n_components=3).fit_transform(Xt).reshape(
        len(uniq_conds), n_tbins, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i, c in enumerate(uniq_conds):
        ax.plot(tr_bins[:-1] + tr_bin / 2, pca_t[i, :, 0], lw=2,
                color=color_map[c], label=c)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("PC1 score")
    ax.set_title("PC1 population trajectory by condition", fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[1]
    for i, c in enumerate(uniq_conds):
        ax.plot(pca_t[i, :, 0], pca_t[i, :, 1], lw=2, color=color_map[c], alpha=0.85)
        for b in range(0, n_tbins, 5):
            ax.scatter(pca_t[i, b, 0], pca_t[i, b, 1], s=18, color=color_map[c],
                       edgecolors="white", linewidths=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Population state-space trajectories", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig05b_time_resolved_pca.png")

    # ---------------- summary ----------------
    summary = {
        "session": str(SESSION),
        "n_clusters": data["n_clusters"],
        "n_anatomical": step1_summary["n_anatomical"],
        "n_quality": step1_summary["n_quality"],
        "n_functional": step1_summary["n_functional"],
        "n_grating_trials": int(grating_df["included"].sum()),
        "n_conditions": int(grating_df["condition"].nunique()),
        "n_sparse_noise_flashes": int(len(flashes)),
        "sparse_grid": f"{grid['n_x']}x{grid['n_y']}",
        "n_sig_regression": n_sig,
        "side_decoding_acc": acc_mean,
        "n_sig_rf": n_rf,
        "pca_evr_pc1": evr[0],
        "pca_evr_pc1_2": pca_res["cum_evr"][1],
        "discriminability": pca_res["discriminability"],
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "part2_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print("\nAll outputs written to: part2_outputs/")


if __name__ == "__main__":
    main()