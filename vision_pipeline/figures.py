"""All figures produced by the pipeline.

Keeping plotting separate from analysis means every figure is a pure function of
a result object, which makes figures cheap to regenerate and easy to test with
synthetic inputs. Figure numbers follow the original ``PART2.py`` naming so
earlier outputs stay comparable; new stages use new numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import AnalysisConfig
from .utils import save_fig

sns.set_theme(style="ticks", context="notebook")


# ---------------------------------------------------------------------------
# Part 2 - stimulus representations
# ---------------------------------------------------------------------------
def plot_stimulus_representations(design, grating_df, occupancy, grid, flashes,
                                  out_dir, include_mask=None):
    """Figure 03: grating design matrix, sparse-noise occupancy, flash sequence."""
    out_dir = Path(out_dir)
    if include_mask is None:
        include_mask = grating_df["included"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(design.loc[include_mask].values.T, aspect="auto",
                         cmap="viridis")
    axes[0].set_yticks(range(design.shape[1]))
    axes[0].set_yticklabels(design.columns)
    axes[0].set_xlabel("Trial (included)")
    axes[0].set_title("Grating design matrix", fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    axes[1].imshow(occupancy.reshape(grid.n_y, grid.n_x).T, origin="lower",
                   aspect="auto", cmap="magma")
    axes[1].set_xlabel("Y position (grid)")
    axes[1].set_ylabel("X position (grid)")
    axes[1].set_title("Sparse-noise occupancy", fontweight="bold")

    sub = flashes[flashes["time"] < flashes["time"].min() + 30]
    axes[2].scatter(sub["time"], sub["y"], s=1, color="black", alpha=0.3)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Y position")
    axes[2].set_title("First 30 s of sparse-noise flashes", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig03_stimulus_representations.png")


# ---------------------------------------------------------------------------
# Part 2 - stimulus-response relationships
# ---------------------------------------------------------------------------
def plot_contrast_tuning(tuning_result, grating_df, out_dir):
    """Figure 04: population contrast tuning and the (left, right) surface."""
    out_dir = Path(out_dir)
    order = tuning_result.order
    conds_order = [tuning_result.conditions[i] for i in order]
    means = tuning_result.matrix[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pop_mean = means.mean(axis=1)
    pop_sem = means.std(axis=1) / np.sqrt(means.shape[1])
    axes[0].errorbar(range(len(conds_order)), pop_mean, yerr=pop_sem, fmt="o-",
                     color="#08519c", capsize=3)
    axes[0].set_xticks(range(len(conds_order)))
    axes[0].set_xticklabels(conds_order, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Population mean response (spikes/window)")
    axes[0].set_title("Population contrast tuning", fontweight="bold")

    cl_vals = np.sort(grating_df["contrast_left"].unique())
    cr_vals = np.sort(grating_df["contrast_right"].unique())
    surface = np.full((len(cl_vals), len(cr_vals)), np.nan)
    for i, cond in enumerate(conds_order):
        cl, cr = _parse(cond)
        surface[np.where(cl_vals == cl)[0][0], np.where(cr_vals == cr)[0][0]] = \
            means[i].mean()

    im1 = axes[1].imshow(surface, origin="lower", aspect="auto", cmap="YlOrRd")
    axes[1].set_xticks(range(len(cr_vals)))
    axes[1].set_xticklabels(cr_vals)
    axes[1].set_yticks(range(len(cl_vals)))
    axes[1].set_yticklabels(cl_vals)
    axes[1].set_xlabel("Right contrast")
    axes[1].set_ylabel("Left contrast")
    axes[1].set_title("Population tuning surface", fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig04_contrast_tuning.png")


def _parse(label):
    left, right = str(label).split()
    return float(left[1:]), float(right[1:])


def plot_regression(reg, units_df, out_dir):
    """Figure 04b: encoding strength and side preference vs depth."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(reg["r2"].dropna(), bins=25, color="#3182bd", edgecolor="white",
                 label="in-sample R²")
    if "r2_cv" in reg:
        axes[0].hist(reg["r2_cv"].dropna(), bins=25, color="#e6550d", alpha=0.6,
                     edgecolor="white", label="cross-validated R²")
    axes[0].axvline(reg["r2"].median(), color="black", ls="--",
                    label=f"median = {reg['r2'].median():.3f}")
    axes[0].set_xlabel("R² (stimulus regression)")
    axes[0].set_ylabel("Number of units")
    axes[0].set_title("Stimulus encoding strength", fontweight="bold")
    axes[0].legend(fontsize=8)

    good = units_df.set_index("cluster_id").loc[reg["cluster_id"]]
    axes[1].scatter(good["depth"].values, reg["side_preference"], s=12, alpha=0.5,
                    c=np.where(reg["significant"], "#e6550d", "#9ecae1"))
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Depth (µm)")
    axes[1].set_ylabel("Side preference  β(R) − β(L)")
    axes[1].set_title("Contrast-side preference vs depth", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig04b_stimulus_regression.png")


def plot_receptive_fields(rf_result, unit_ids, out_dir, n_examples=6):
    """Figures 04c/04d: example RFs and the significance fraction."""
    out_dir = Path(out_dir)
    grid = rf_result.grid
    from .receptive_fields import example_units

    examples = example_units(rf_result, n=n_examples, only_significant=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, uid in zip(axes, examples):
        zmap = grid.reshape_map(rf_result.maps[int(uid)]["z"])
        vmax = np.nanmax(np.abs(zmap)) or 1.0
        ax.imshow(zmap, origin="lower", aspect="auto", cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax)
        ax.set_title(f"unit {uid}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(examples):]:
        ax.axis("off")
    fig.suptitle("Example sparse-noise receptive fields (z-scored STA)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir / "fig04c_receptive_fields.png")

    table = rf_result.table
    frac_perm = 100 * table["significant_perm"].mean()
    frac_gauss = 100 * table["significant"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["permutation\n(primary)", "Gaussian max|z|\n(legacy)"],
           [frac_perm, frac_gauss], color=["#74c476", "#9ecae1"],
           edgecolor="white")
    ax.axhline(5, color="black", ls="--", lw=0.8, label="5% expected by chance")
    ax.set_ylabel("% units with significant RF")
    ax.set_ylim(0, 105)
    ax.set_title(f"RF significance (threshold max|z| = "
                 f"{rf_result.z_threshold_gaussian:.1f})", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig04d_rf_fraction.png")


# ---------------------------------------------------------------------------
# Part 2 - population
# ---------------------------------------------------------------------------
def condition_color_map(conditions):
    uniq = list(dict.fromkeys(list(conditions)))
    palette = sns.color_palette("husl", max(len(uniq), 1))
    return dict(zip(uniq, palette)), uniq


def plot_pca(pca_result, conditions, out_dir):
    """Figure 05: scree plot, PC scatter, within/across condition distances."""
    out_dir = Path(out_dir)
    evr = pca_result.evr
    scores = pca_result.scores
    color_map, uniq = condition_color_map(conditions)

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 1])

    ax = fig.add_subplot(gs[0])
    ax.plot(np.arange(1, len(evr) + 1), evr * 100, "o-", color="#08519c")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Scree plot", fontweight="bold")

    ax = fig.add_subplot(gs[1])
    conditions = np.asarray(conditions)
    for c in uniq:
        m = conditions == c
        ax.scatter(scores[m, 0], scores[m, 1], s=22, alpha=0.75, color=color_map[c],
                   label=c, edgecolors="white", linewidths=0.3)
        if m.sum():
            ax.scatter(scores[m, 0].mean(), scores[m, 1].mean(), marker="X", s=160,
                       color="black", edgecolors="white", linewidths=0.8)
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)" if len(evr) > 1 else "PC2")
    ax.set_title("Population responses in PC space", fontweight="bold")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))

    ax = fig.add_subplot(gs[2])
    ax.bar(["within", "across"], [pca_result.within_mean, pca_result.across_mean],
           color=["#6baed6", "#fb6a4a"], edgecolor="white")
    ax.set_ylabel("Mean distance (PC1-3)")
    ax.set_title(f"Discriminability = {pca_result.discriminability:.2f}",
                 fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig05_pca.png")


def plot_trajectories(centers, traj, labels, color_map, bin_size, out_dir):
    """Figure 05b: PC1 trajectories and population state-space paths."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i, c in enumerate(labels):
        ax.plot(centers, traj[i, :, 0], lw=2, color=color_map[c], label=c)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("PC1 score")
    ax.set_title("PC1 population trajectory by condition", fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[1]
    n_tbins = traj.shape[1]
    for i, c in enumerate(labels):
        ax.plot(traj[i, :, 0], traj[i, :, 1], lw=2, color=color_map[c], alpha=0.85)
        for b in range(0, n_tbins, 5):
            ax.scatter(traj[i, b, 0], traj[i, b, 1], s=18, color=color_map[c],
                       edgecolors="white", linewidths=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Population state-space trajectories", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig05b_time_resolved_pca.png")


# ---------------------------------------------------------------------------
# Model-based stage
# ---------------------------------------------------------------------------
def plot_glm_summary(glm_result, out_dir):
    """Figure 06: GLM fit quality and example stimulus kernels."""
    out_dir = Path(out_dir)
    table = glm_result.table
    ok = table["fit"].fillna(False).astype(bool)
    if not ok.any():
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(table.loc[ok, "r2_dev"].dropna(), bins=20, alpha=0.7,
                 color="#3182bd", edgecolor="white", label="in-sample")
    axes[0].hist(table.loc[ok, "r2_dev_cv"].dropna(), bins=20, alpha=0.7,
                 color="#e6550d", edgecolor="white", label="held-out")
    axes[0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Deviance explained (pseudo-R²)")
    axes[0].set_ylabel("Number of units")
    axes[0].set_title("Poisson GLM fit quality", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].scatter(table.loc[ok, "n_spikes"], table.loc[ok, "r2_dev_cv"], s=18,
                    alpha=0.7, c=np.where(table.loc[ok, "significant"],
                                           "#e6550d", "#9ecae1"))
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Total spikes on the sparse-noise period")
    axes[1].set_ylabel("Held-out deviance explained")
    axes[1].set_title("Encoding quality vs spike count", fontweight="bold")

    # example kernels: highest held-out deviance
    best = table.loc[ok].sort_values("r2_dev_cv", ascending=False)["cluster_id"]
    n_show = min(4, len(best))
    for uid in best.head(n_show):
        k = glm_result.filters.get(int(uid))
        if k is None:
            continue
        axes[2].plot(k, alpha=0.8, label=f"unit {int(uid)}")
    axes[2].axhline(0, color="black", lw=0.8, ls="--")
    axes[2].set_xlabel("Stimulus-cell coefficient index")
    axes[2].set_ylabel("Stimulus kernel weight")
    axes[2].set_title("Stimulus filters (top units)", fontweight="bold")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig06_glm_encoding.png")


def plot_glm_filter_maps(glm_result, grid, out_dir, n_examples=4):
    """Figure 06b: stimulus kernels shown on the sparse-noise grid."""
    out_dir = Path(out_dir)
    table = glm_result.table
    ok = table["fit"].fillna(False).astype(bool)
    if not ok.any():
        return None
    best = table.loc[ok].sort_values("r2_dev_cv", ascending=False)["cluster_id"]
    examples = [int(u) for u in best.head(n_examples) if int(u) in glm_result.filters]
    if not examples:
        return None

    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4),
                             squeeze=False)
    for ax, uid in zip(axes.ravel(), examples):
        k = glm_result.filters[uid]
        vmax = np.nanmax(np.abs(k)) or 1.0
        ax.imshow(grid.reshape_map(k), origin="lower", aspect="auto",
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"unit {uid}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Poisson GLM stimulus kernels (RF estimates)", fontweight="bold",
                 y=1.02)
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig06b_glm_kernels.png")


def plot_glm_sta_agreement(agreement, out_dir):
    """Figure 06c: agreement between GLM kernels and STA maps."""
    out_dir = Path(out_dir)
    if agreement is None or agreement.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(agreement["kernel_sta_corr"].dropna(), bins=20, color="#756bb1",
            edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Correlation(GL M kernel, STA map) per unit")
    ax.set_ylabel("Number of units")
    ax.set_title("Encoding model vs spike-triggered average", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig06c_glm_vs_sta.png")


def plot_rrr_curve(curve, out_dir):
    """Figure 07: reduced-rank regression performance vs rank."""
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(curve["rank"], curve["r2"], "o-", color="#08519c", label="in-sample R²")
    if "r2_cv" in curve:
        ax.plot(curve["rank"], curve["r2_cv"], "s--", color="#e6550d",
                label="cross-validated R²")
    full = curve.attrs.get("r2_full")
    if full is not None and np.isfinite(full):
        ax.axhline(full, color="grey", ls=":", lw=1,
                   label=f"full-rank OLS R² = {full:.3f}")
    ax.set_xlabel("Rank of the stimulus→population map")
    ax.set_ylabel("R²")
    ax.set_title("Reduced-rank regression: stimulus → population", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig07_rrr_rank_curve.png")


def plot_latent_models(fa_result, var_fit, eig_summary, dim_selection, out_dir):
    """Figure 08: factor-analysis loadings, VAR eigenvalues, model selection."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(fa_result.explained, bins=20, color="#31a354", edgecolor="white")
    axes[0].set_xlabel("Fraction of unit variance explained by shared factors")
    axes[0].set_ylabel("Number of units")
    axes[0].set_title(
        f"Factor analysis ({fa_result.loadings.shape[1]} factors, "
        f"{fa_result.total_explained:.2f} total)", fontweight="bold")

    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", ls=":", lw=1)
    if len(eig_summary):
        ax.scatter(eig_summary["real"], eig_summary["imag"], s=45,
                   color="#e6550d", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title("VAR(1) latent eigenvalues", fontweight="bold")

    ax = axes[2]
    if dim_selection is not None and not dim_selection.empty:
        d = dim_selection.dropna(subset=["aic"])
        if len(d):
            ax.plot(d["n_factors"], d["aic"], "o-", color="#08519c", label="AIC")
            ax.plot(d["n_factors"], d["bic"], "s--", color="#e6550d", label="BIC")
            best = int(d.loc[d["bic"].idxmin(), "n_factors"])
            ax.axvline(best, color="black", ls=":", lw=1,
                       label=f"BIC best = {best}")
        else:
            ax.text(0.5, 0.5, "dynamic factor model\ndid not converge",
                    ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Number of latent factors")
    ax.set_ylabel("Information criterion")
    ax.set_title("Latent dimension selection", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig08_latent_models.png")


# ---------------------------------------------------------------------------
# Multi-session stage
# ---------------------------------------------------------------------------
def plot_multisession_overview(df, out_dir):
    """Figure 10: per-session and per-subject summary of the pipeline outputs."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    d = df.sort_values("n_functional", ascending=False)
    ax.bar(range(len(d)), d["n_functional"], color="#3182bd", edgecolor="white")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["session_label"], rotation=90, fontsize=6)
    ax.set_ylabel("Refined visual units")
    ax.set_title("Visual population size per session", fontweight="bold")

    ax = axes[0, 1]
    ax.hist(df["responsiveness_rate"].dropna(), bins=15, color="#74c476",
            edgecolor="white")
    ax.set_xlabel("Fraction of good visual-area units that are responsive")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Functional drive across sessions", fontweight="bold")

    ax = axes[1, 0]
    ax.hist(df["side_decoding_acc"].dropna(), bins=15, color="#756bb1",
            edgecolor="white")
    ax.axvline(0.5, color="black", ls="--", lw=1, label="chance")
    ax.set_xlabel("Side-decoding accuracy")
    ax.set_ylabel("Number of sessions")
    ax.set_title("Stimulus decoding across sessions", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for subject, sub in df.groupby("subject"):
        ax.scatter(sub["n_functional"], sub["rf_fraction"], s=28, alpha=0.8,
                   label=subject)
    ax.set_xlabel("Refined visual units")
    ax.set_ylabel("Fraction with significant RF")
    ax.set_title("Population size vs receptive-field coverage", fontweight="bold")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig10_multisession_overview.png")


def plot_multisession_relationships(df, out_dir):
    """Figure 11: cross-session relationships between summary metrics."""
    out_dir = Path(out_dir)
    cols = ["n_quality", "n_functional", "responsiveness_rate",
            "side_decoding_acc", "rf_fraction", "pca_evr_pc1",
            "mean_visual_modulation"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) < 3:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    corr = sub.corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Between-session metric correlations", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig11_multisession_relationships.png")


def plot_multisession_by_subject(df, out_dir):
    """Figure 12: metric distributions per subject."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("n_functional", "Refined visual units"),
               ("side_decoding_acc", "Side-decoding accuracy"),
               ("rf_fraction", "Significant-RF fraction")]
    for ax, (col, label) in zip(axes, metrics):
        if col not in df:
            continue
        sns.boxplot(data=df, x="subject", y=col, ax=ax, color="#c6dbef")
        sns.stripplot(data=df, x="subject", y=col, ax=ax, color="black", size=3,
                      alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        if col == "side_decoding_acc":
            ax.axhline(0.5, color="black", ls="--", lw=1)
    fig.suptitle("Cross-subject variability", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, out_dir / "fig12_multisession_by_subject.png")
