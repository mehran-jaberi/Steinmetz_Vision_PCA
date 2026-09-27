"""Refined visual-neuron identification (anatomical -> quality -> functional).

The three-stage cascade implemented here is the entry point of the visual
pipeline:

1. **Anatomical** - each unit is assigned a brain region through its peak
   recording channel (``channels.brainLocation.tsv`` / Allen ontology) and
   restricted to visual areas (``VIS*``).
2. **Quality** - only well-isolated units are kept
   (``clusters._phy_annotation == 1``, i.e. "good" units) with a minimum total
   spike count.
3. **Functional** - units must be significantly driven by task-grating stimulus
   onset (paired t-test across stimulus trials, Benjamini-Hochberg FDR).

The functional stage uses the *task gratings* rather than the passive-visual
block: the passive block did not reliably drive units in this dataset, and the
grating stimulus is the one used by the downstream stimulus-response analyses.
This choice is documented here because it directly affects which units enter the
population.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import VISUAL_AREAS, AnalysisConfig
from .utils import bh_fdr, count_spikes_in_windows, save_fig


@dataclass
class VisualNeuronResult:
    """Outcome of the refinement cascade."""

    units: pd.DataFrame          # one row per cluster
    refined_ids: np.ndarray      # cluster ids passing all three stages
    summary: dict                # stage counts
    stim_times: np.ndarray       # trials used for the functional stage


def build_unit_table(session) -> pd.DataFrame:
    """Metadata table with one row per cluster (all clusters, no filtering)."""
    region = session.channel_region[session.cluster_peak_channel]
    probe_names = session.probe_names
    n = session.n_clusters
    n_spikes_per_cluster = session.n_spikes_per_cluster

    return pd.DataFrame(
        {
            "cluster_id": np.arange(n),
            "probe": session.cluster_probes.astype(int),
            "probe_name": [probe_names[int(p)] for p in session.cluster_probes],
            "region": region,
            "depth": session.cluster_depths,
            "peak_channel": session.cluster_peak_channel.astype(int),
            "quality": session.cluster_annotation.astype(int),
            "n_spikes": n_spikes_per_cluster.astype(int),
        }
    )


def stimulus_trials(session, require_nonzero_contrast=True):
    """Trial times used for the functional (responsiveness) stage."""
    tr = session.trial_info
    times = np.asarray(tr["visualStim_times"], float)
    if not require_nonzero_contrast:
        return times
    has_stim = (tr["visualStim_contrastLeft"] > 0) | (
        tr["visualStim_contrastRight"] > 0
    )
    return times[np.asarray(has_stim, bool)]


def responsiveness(session, unit_ids, stim_times, stim_window, base_window):
    """Paired stimulus-vs-baseline comparison for a set of units.

    Returns a dict of arrays (one entry per unit) with mean rates in the
    stimulus and baseline windows, the modulation index, and the paired t-test
    t/p values.
    """
    unit_ids = np.asarray(unit_ids, dtype=int)
    n = len(unit_ids)
    stim_dur = stim_window[1] - stim_window[0]
    base_dur = base_window[1] - base_window[0]

    resp_rates = np.zeros(n)
    base_rates = np.zeros(n)
    tvals = np.full(n, np.nan)
    pvals = np.full(n, np.nan)
    mods = np.zeros(n)

    for k, uid in enumerate(unit_ids):
        s = session.unit_spike_times(int(uid))
        r = count_spikes_in_windows(s, stim_times, stim_window) / stim_dur
        b = count_spikes_in_windows(s, stim_times, base_window) / base_dur
        resp_rates[k] = r.mean()
        base_rates[k] = b.mean()
        mods[k] = (r.mean() - b.mean()) / (r.mean() + b.mean() + 1e-6)
        if r.size > 1 and np.any(r != b):
            try:
                t, p = stats.ttest_rel(r, b)
            except Exception:
                t, p = np.nan, np.nan
        else:
            t, p = np.nan, np.nan
        tvals[k], pvals[k] = t, p

    return {
        "resp_rate": resp_rates,
        "base_rate": base_rates,
        "modulation": mods,
        "t": tvals,
        "p": pvals,
    }


def identify_visual_neurons(session, cfg: AnalysisConfig | None = None,
                            verbose=True) -> VisualNeuronResult:
    """Run the anatomical -> quality -> functional refinement cascade."""
    cfg = cfg or AnalysisConfig()
    units = build_unit_table(session)

    # ---- Stage A: anatomical ----
    units["in_visual_area"] = units["region"].isin(VISUAL_AREAS)
    n_anatomical = int(units["in_visual_area"].sum())

    # ---- Stage B: quality ----
    units["is_good"] = units["quality"] == cfg.good_annotation
    units["passes_basic"] = (
        units["in_visual_area"]
        & units["is_good"]
        & (units["n_spikes"] >= cfg.min_total_spikes)
    )
    n_quality = int(units["passes_basic"].sum())

    # ---- Stage C: functional ----
    stim_times = stimulus_trials(session)
    candidates = units.index[units["passes_basic"]].tolist()
    candidate_ids = units.loc[candidates, "cluster_id"].values

    resp = responsiveness(
        session, candidate_ids, stim_times, cfg.stim_window, cfg.base_window
    )

    # FDR correction across the candidate set only
    pvals_ok = ~np.isnan(resp["p"])
    sig = np.zeros(len(candidates), dtype=bool)
    if np.any(pvals_ok):
        sig[pvals_ok] = bh_fdr(resp["p"][pvals_ok], alpha=cfg.fdr_alpha)

    # pre-declare the columns so non-candidates get explicit NaN/False rather
    # than an object-dtype mixture (which silently downcasts later on)
    for col in ("visual_resp_rate", "visual_base_rate", "visual_t", "visual_p",
                "visual_modulation"):
        units[col] = np.nan
    units["visually_responsive"] = False

    units.loc[candidates, "visual_resp_rate"] = resp["resp_rate"]
    units.loc[candidates, "visual_base_rate"] = resp["base_rate"]
    units.loc[candidates, "visual_t"] = resp["t"]
    units.loc[candidates, "visual_p"] = resp["p"]
    units.loc[candidates, "visual_modulation"] = resp["modulation"]
    units.loc[candidates, "visually_responsive"] = sig

    units["passes_functional"] = (
        units["passes_basic"] & units["visually_responsive"]
    )

    # baseline-corrected response and firing-rate extras (documentation value)
    units["visual_delta_rate"] = units["visual_resp_rate"] - units["visual_base_rate"]
    units["firing_rate_hz"] = units["n_spikes"] / max(session.duration, 1e-9)
    units["refined"] = units["passes_functional"]

    # backwards-compatible column name used by the original pipeline
    units["visual_refined"] = units["refined"]

    refined_ids = units.loc[units["refined"], "cluster_id"].values.astype(int)

    summary = {
        "n_total": session.n_clusters,
        "n_anatomical": n_anatomical,
        "n_quality": n_quality,
        "n_functional": int(units["refined"].sum()),
        "n_stim_trials": int(len(stim_times)),
        "responsiveness_rate": (
            float(units["passes_functional"].sum() / max(n_quality, 1))
        ),
    }

    if verbose:
        print("\n[Stage 1] Refined visual-neuron identification")
        print(
            f"  {n_anatomical} units in visual areas -> {n_quality} good-quality "
            f"-> {summary['n_functional']} visually responsive (FDR p<{cfg.fdr_alpha})"
        )

    return VisualNeuronResult(units=units, refined_ids=refined_ids,
                              summary=summary, stim_times=stim_times)


def plot_refinement(result: VisualNeuronResult, out_dir, cfg: AnalysisConfig | None
                    = None):
    """Figures 01/01b: refinement cascade, region composition, depth profile."""
    cfg = cfg or AnalysisConfig()
    units = result.units
    s = result.summary

    # (a) refinement cascade + (b) region composition
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    stage_names = [
        "Anatomical\n(VIS areas)",
        "+ Quality\n(good units)",
        "+ Responsive\n(grating onset)",
    ]
    stage_counts = [s["n_anatomical"], s["n_quality"], s["n_functional"]]
    axes[0].bar(
        stage_names, stage_counts,
        color=["#6baed6", "#3182bd", "#08519c"], edgecolor="white",
    )
    for i, c in enumerate(stage_counts):
        axes[0].text(i, c + 0.5, str(c), ha="center", fontsize=11,
                     fontweight="bold")
    axes[0].set_ylabel("Number of units")
    axes[0].set_title("Refinement cascade", fontweight="bold")

    comp = units.loc[units["refined"], "region"].value_counts()
    axes[1].bar(range(len(comp)), comp.values, color="#74c476", edgecolor="white")
    axes[1].set_xticks(range(len(comp)))
    axes[1].set_xticklabels(comp.index, rotation=45, ha="right")
    axes[1].set_ylabel("Number of units")
    axes[1].set_title("Refined visual set: region composition", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir / "fig01_visual_neuron_refinement.png")

    # (c) depth profile with responsiveness overlay
    vis = units[units["in_visual_area"]]
    responsive = vis["visually_responsive"].fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        vis["depth"], vis["visual_modulation"], s=14, alpha=0.55,
        c=np.where(responsive, "#e6550d", "#9ecae1"),
    )
    ax.set_xlabel("Depth along probe (µm)")
    ax.set_ylabel("Grating modulation index")
    ax.set_title("Visual-area units: responsiveness vs depth", fontweight="bold")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", ls="", color="#e6550d",
                       label="responsive"),
            plt.Line2D([0], [0], marker="o", ls="", color="#9ecae1",
                       label="not responsive"),
        ],
        loc="best",
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "fig01b_responsiveness_vs_depth.png")


def responses_by_decile(units, column="visual_modulation", n_bins=10):
    """Depth-binned summary of a per-unit metric (used by cross-session plots)."""
    vis = units[units["in_visual_area"]].copy()
    if vis.empty:
        return pd.DataFrame(columns=["depth_bin", "mean", "n"])
    vis["depth_bin"] = pd.qcut(vis["depth"], n_bins, duplicates="drop")
    return (
        vis.groupby("depth_bin", observed=True)[column]
        .agg(["mean", "count"])
        .rename(columns={"count": "n"})
        .reset_index()
    )