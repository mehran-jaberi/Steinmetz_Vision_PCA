"""Stage orchestration: single-session, validation, model-based, multi-session.

Each ``run_*`` function is a thin, explicit sequence of the analysis functions
in the other modules plus writing tables/figures and a provenance manifest.
Keeping orchestration in one place makes the pipeline order auditable, and makes
it possible to call any stage independently (which is what the CLI does).
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as fig
from .config import AnalysisConfig
from .encoding import (compute_contrast_tuning, decode_side, decode_side_null,
                       regress_responses_on_stimulus)
from .glm import fit_population_glms, glm_vs_sta_agreement
from .io import SessionRef, discover_sessions, load_session_alf
from .latent import (eigenvalue_summary, factor_analysis, fit_var1,
                     rrr_rank_curve, select_latent_dimension)
from .population import population_pca, time_resolved_pca
from .receptive_fields import compute_receptive_fields
from .reporting import write_manifest
from .stimuli import (build_stimulus_movie, extract_grating_representations,
                      reconstruct_grating_stimuli, reconstruct_sparse_noise,
                      sparse_noise_occupancy)
from .units import identify_visual_neurons, plot_refinement
from .utils import write_table
from .validation import validate_single_session


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Part2Result:
    """Everything the single-session stage produces."""

    session: object
    cfg: AnalysisConfig
    units: object
    grating: pd.DataFrame
    flashes: pd.DataFrame
    grid: object
    design: pd.DataFrame
    movie_edges: np.ndarray
    movie: np.ndarray
    occupancy: np.ndarray
    tuning: object
    reg: pd.DataFrame
    rf: object
    pca: object
    centers: np.ndarray
    traj: np.ndarray
    traj_labels: list
    traj_evr: np.ndarray
    decode_mean: float
    decode_std: float
    decode_null_mean: float
    decode_null_p95: float
    decode_n: int
    summary: pd.DataFrame = field(default=None, repr=False)

    @property
    def refined_ids(self):
        return self.units.refined_ids

    @property
    def conditions(self):
        return self.tuning.conds


@dataclass
class ModelResult:
    glm: object
    rrr: pd.DataFrame
    factor: object
    var: dict
    eigenvalues: pd.DataFrame
    dim_selection: pd.DataFrame
    agreement: pd.DataFrame
    summary: pd.DataFrame = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Stage 1: single session (Part 2 as published, plus refinements)
# ---------------------------------------------------------------------------
def run_part2(session, out_dir, cfg: AnalysisConfig | None = None,
              make_figures=True, n_perm_reg=0, verbose=True) -> Part2Result:
    """Run the complete single-session vision pipeline."""
    cfg = cfg or AnalysisConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: identification ----
    if verbose:
        print("\n" + "#" * 78)
        print("# STAGE 1: REFINED VISUAL-NEURON IDENTIFICATION")
        print("#" * 78)
    units = identify_visual_neurons(session, cfg, verbose=verbose)
    if make_figures:
        plot_refinement(units, out_dir, cfg)
    write_table(units.units, out_dir / "visual_units.csv", verbose=verbose)
    refined_ids = units.refined_ids
    if verbose:
        print(f"  Refined visual population: {len(refined_ids)} units")

    # ---- Stage 2: stimulus reconstruction ----
    if verbose:
        print("\n" + "#" * 78)
        print("# STAGE 2: VISUAL STIMULUS RECONSTRUCTION")
        print("#" * 78)
    grating = reconstruct_grating_stimuli(session)
    flashes, grid = reconstruct_sparse_noise(session)
    write_table(grating, out_dir / "grating_stimuli.csv", verbose=verbose)
    write_table(flashes, out_dir / "flashes.csv", verbose=verbose)
    if verbose:
        print(f"  Grating trials        : {len(grating)} "
              f"(included: {int(grating['included'].sum())})")
        print(f"  Stimulus conditions   : {grating['condition'].nunique()}")
        print(f"  Sparse-noise flashes  : {len(flashes)} on a "
              f"{grid.n_x} x {grid.n_y} grid")

    # ---- Stage 3: stimulus representations ----
    if verbose:
        print("\n" + "#" * 78)
        print("# STAGE 3: QUANTITATIVE STIMULUS REPRESENTATIONS")
        print("#" * 78)
    design = extract_grating_representations(grating)
    write_table(design, out_dir / "stimulus_design.csv", verbose=verbose)
    movie_edges, movie = build_stimulus_movie(flashes, grid, bin_size=cfg.movie_bin)
    occupancy = sparse_noise_occupancy(flashes, grid)
    if verbose:
        print(f"  Grating design matrix : {design.shape[1]} features x "
              f"{design.shape[0]} trials")
        print(f"  Stimulus movie        : {movie.shape[0]} time bins x "
              f"{movie.shape[1]} cells")
        print(f"  Sparse-noise occupancy: {int(occupancy.min())}-"
              f"{int(occupancy.max())} flashes per cell")
    if make_figures:
        fig.plot_stimulus_representations(design, grating, occupancy, grid,
                                          flashes, out_dir)

    if len(refined_ids) < cfg.min_units:
        raise RuntimeError(
            f"refined visual population too small ({len(refined_ids)} < "
            f"{cfg.min_units}); aborting downstream analysis"
        )

    # ---- Stage 4: stimulus -> activity ----
    if verbose:
        print("\n" + "#" * 78)
        print("# STAGE 4: RELATING STIMULUS REPRESENTATIONS TO NEURAL ACTIVITY")
        print("#" * 78)
    tuning = compute_contrast_tuning(session, refined_ids, grating, cfg)
    tidy = pd.DataFrame(tuning.matrix.T, columns=tuning.conditions)
    tidy.insert(0, "cluster_id", refined_ids)
    write_table(tidy, out_dir / "tuning.csv", verbose=verbose)
    if verbose:
        print(f"  Contrast tuning: {len(tuning.conditions)} conditions x "
              f"{len(refined_ids)} units")
    if make_figures:
        fig.plot_contrast_tuning(tuning, grating, out_dir)

    # reuse the response matrix computed by the tuning stage
    reg = regress_responses_on_stimulus(session, refined_ids, grating, design,
                                        cfg, X=tuning.X, n_perm=n_perm_reg,
                                        verbose=verbose)
    write_table(reg, out_dir / "regression.csv", verbose=verbose)
    if make_figures:
        fig.plot_regression(reg, units.units, out_dir)

    use_mask = grating["included"].values & (grating["abs_contrast"].values > 0)
    decode_mean, decode_std, decode_n, y_side, X_side = decode_side(
        tuning.X, grating, use_mask, cfg
    )
    null_mean, null_p95, _ = decode_side_null(X_side, y_side, cfg)
    if verbose:
        print(f"  Side decoding (unilateral trials, n={decode_n}): "
              f"{decode_mean * 100:.1f} ± {decode_std * 100:.1f} % "
              f"(permutation null {null_mean * 100:.1f}%, p95 "
              f"{null_p95 * 100:.1f}%)")

    rf = compute_receptive_fields(session, refined_ids, flashes, grid, cfg,
                                  verbose=verbose)
    # Robustness check: the same STA restricted to temporally isolated
    # presentations. The sparse-noise sequence runs at a minimum ISI (~10 ms)
    # far shorter than the STA window, so neighbouring flashes contaminate the
    # all-flash average; this quantifies how much RF evidence survives when
    # that contamination is removed. The all-flash result stays the primary
    # analysis so earlier outputs remain reproducible.
    if not cfg.rf_isolated_only:
        rf_iso = compute_receptive_fields(
            session, refined_ids, flashes, grid,
            cfg.with_overrides(rf_isolated_only=True), verbose=False,
        )
        rf.table["z_max_iso"] = rf_iso.table["z_max"].values
        rf.table["p_perm_iso"] = rf_iso.table["p_perm"].values
        rf.table["significant_perm_iso"] = rf_iso.table[
            "significant_perm"].values
        if verbose:
            print(f"  RF robustness (isolated flashes only, "
                  f"n={rf_iso.n_flashes_used}/{rf.n_flashes_total}): "
                  f"{int(rf_iso.table['significant_perm'].sum())}/"
                  f"{len(rf_iso.table)} units significant vs "
                  f"{int(rf.table['significant_perm'].sum())}/"
                  f"{len(rf.table)} on all flashes")
    write_table(rf.table, out_dir / "receptive_fields.csv", verbose=verbose)
    if make_figures:
        fig.plot_receptive_fields(rf, refined_ids, out_dir)

    # ---- Stage 5: population / PCA ----
    if verbose:
        print("\n" + "#" * 78)
        print("# STAGE 5: PCA / DIMENSIONALITY REDUCTION")
        print("#" * 78)
    X_pop = tuning.X
    pca = population_pca(X_pop, tuning.conds, cfg)
    evr = pca.evr
    if verbose:
        print(f"  PCA on {X_pop.shape[0]} trials x {X_pop.shape[1]} units")
        print(f"  Explained variance: PC1 = {evr[0] * 100:.1f}%, "
              f"PC1-2 = {pca.cum_evr[1] * 100:.1f}%, "
              f"PC1-5 = {evr[:5].sum() * 100:.1f}%")
        print(f"  Condition discriminability (1 - within/across distance) "
              f"= {pca.discriminability:.3f}")
    if make_figures:
        fig.plot_pca(pca, tuning.conds, out_dir)

    centers, traj, traj_evr, traj_labels, _ = time_resolved_pca(
        session, refined_ids, grating.loc[use_mask, "onset"].values,
        tuning.conds, cfg
    )
    if make_figures:
        color_map, _ = fig.condition_color_map(traj_labels)
        fig.plot_trajectories(centers, traj, traj_labels, color_map,
                              cfg.traj_bin, out_dir)

    # per-unit PC loadings (written as a table, not stored on the result)
    write_table(pca.component_weights(refined_ids),
                out_dir / "pca_loadings.csv", verbose=verbose)
    # ---- summary ----
    summary = _build_summary(session, units, grating, flashes, grid, reg,
                            decode_mean, rf, evr, pca, cfg)
    write_table(summary, out_dir / "part2_summary.csv", verbose=verbose)

    result = Part2Result(
        session=session, cfg=cfg, units=units, grating=grating, flashes=flashes,
        grid=grid, design=design, movie_edges=movie_edges, movie=movie,
        occupancy=occupancy, tuning=tuning, reg=reg, rf=rf, pca=pca,
        centers=centers, traj=traj, traj_labels=traj_labels, traj_evr=traj_evr,
        decode_mean=decode_mean, decode_std=decode_std,
        decode_null_mean=null_mean, decode_null_p95=null_p95, decode_n=decode_n,
        summary=summary,
    )

    write_manifest(out_dir / "manifest.json", "part2", cfg,
                   {"session": session.session,
                    "n_refined_units": int(len(refined_ids))})

    if verbose:
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        for k, v in summary.iloc[0].items():
            print(f"  {k:28s}: {v}")
        print(f"\nAll outputs written to: {out_dir}")
    return result


def _build_summary(session, units, grating, flashes, grid, reg, decode_mean, rf,
                   evr, pca, cfg):
    s = units.summary
    row = {
        "session": str(session.session),
        "n_clusters": session.n_clusters,
        "n_anatomical": s["n_anatomical"],
        "n_quality": s["n_quality"],
        "n_functional": s["n_functional"],
        "responsiveness_rate": s["responsiveness_rate"],
        "n_grating_trials": int(grating["included"].sum()),
        "n_conditions": int(grating["condition"].nunique()),
        "n_sparse_noise_flashes": int(len(flashes)),
        "sparse_grid": f"{grid.n_x}x{grid.n_y}",
        "n_sig_regression": int(reg["significant"].sum()),
        "median_r2": float(reg["r2"].median()),
        "median_r2_cv": float(reg["r2_cv"].median()),
        "side_decoding_acc": decode_mean,
        "n_sig_rf": int(rf.table["significant"].sum()),
        "n_sig_rf_permutation": int(rf.table["significant_perm"].sum()),
        "n_sig_rf_permutation_iso": (int(rf.table["significant_perm_iso"].sum())
                                     if "significant_perm_iso" in rf.table
                                     else -1),
        "pca_evr_pc1": float(evr[0]),
        "pca_evr_pc1_2": float(pca.cum_evr[1]),
        "discriminability": pca.discriminability,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Stage 2: model-based population analysis
# ---------------------------------------------------------------------------
def run_model_stage(part2: Part2Result, out_dir, cfg: AnalysisConfig | None = None,
                    make_figures=True, verbose=True) -> ModelResult:
    """Poisson GLM encoding models, reduced-rank regression and latent dynamics."""
    cfg = cfg or AnalysisConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session, refined_ids = part2.session, part2.refined_ids

    if verbose:
        print("\n" + "#" * 78)
        print("# STAGE 6: MODEL-BASED POPULATION ANALYSIS")
        print("#" * 78)

    # ---- (a) Poisson GLM encoding models on the sparse-noise movie ----
    glm = fit_population_glms(session, refined_ids, part2.movie,
                              part2.movie_edges, part2.grid, cfg,
                              verbose=verbose)
    write_table(glm.table, out_dir / "glm_encoding.csv", verbose=verbose)
    agreement = glm_vs_sta_agreement(glm, part2.rf)
    write_table(agreement, out_dir / "glm_vs_sta.csv", verbose=verbose)
    if make_figures:
        fig.plot_glm_summary(glm, out_dir)
        fig.plot_glm_filter_maps(glm, part2.grid, out_dir)
        fig.plot_glm_sta_agreement(agreement, out_dir)

    # ---- (b) reduced-rank regression: stimulus movie -> population ----
    from .responses import population_time_series
    n_bins = part2.movie.shape[0]
    edges = part2.movie_edges
    Y = population_time_series(session, refined_ids, edges)
    lag = 1
    X = part2.movie[: n_bins - lag].astype(float)
    Y = Y[lag:]
    rrr = rrr_rank_curve(Y, X, cfg=cfg)
    write_table(rrr, out_dir / "rrr_rank_curve.csv", verbose=verbose)
    if verbose:
        best = rrr.loc[rrr["r2_cv"].idxmax()] if rrr["r2_cv"].notna().any() else None
        if best is not None:
            print(f"  Reduced-rank regression: best held-out rank = "
                  f"{int(best['rank'])} (CV R² = {best['r2_cv']:.4f}; "
                  f"full-rank in-sample R² = {rrr.attrs['r2_full']:.4f})")
    if make_figures:
        fig.plot_rrr_curve(rrr, out_dir)

    # ---- (c) latent dynamics ----
    # trial-level latents (shared variance) from the response matrix
    factor = factor_analysis(part2.tuning.X, n_factors=3, seed=cfg.random_seed)
    if verbose:
        print(f"  Factor analysis: {factor.loadings.shape[1]} factors explain "
              f"{factor.total_explained * 100:.1f}% of the total response variance "
              f"(median {np.median(factor.explained) * 100:.1f}% per unit)")

    var = fit_var1(part2.traj)
    eigenvalues = eigenvalue_summary(var["eigenvalues"], cfg.traj_bin)
    if verbose:
        print(f"  VAR(1) on {var['n_obs']} latent transitions: one-step R² = "
              f"{var['r2']:.3f}, max |eigenvalue| = "
              f"{eigenvalues['magnitude'].max():.3f}")
    write_table(eigenvalues, out_dir / "latent_var1_eigenvalues.csv",
                verbose=verbose)

    # dynamic factor model on the concatenated condition-averaged activity
    cond_series = _concatenate_conditions(part2)
    dim_selection = select_latent_dimension(cond_series, cfg=cfg)
    write_table(dim_selection, out_dir / "latent_dimension_selection.csv",
                verbose=verbose)
    if verbose:
        n_conv = (int(dim_selection["converged"].sum())
                  if "converged" in dim_selection else 0)
        best_k = _bic_best_k(dim_selection)
        if best_k >= 0:
            print(f"  Dynamic factor model: BIC selects {best_k} latent factors "
                  f"({n_conv}/{len(dim_selection)} fits converged; "
                  f"k = {list(cfg.latent_factors)})")
        else:
            print(f"  Dynamic factor model: no converged fit "
                  f"({n_conv}/{len(dim_selection)}); BIC selection not usable")
    if make_figures:
        fig.plot_latent_models(factor, var, eigenvalues, dim_selection, out_dir)

    # latent factor weights per unit
    loadings = pd.DataFrame(factor.loadings,
                            columns=[f"factor{i + 1}"
                                     for i in range(factor.loadings.shape[1])])
    loadings.insert(0, "cluster_id", refined_ids)
    write_table(loadings, out_dir / "latent_factor_loadings.csv", verbose=verbose)

    summary = pd.DataFrame([{
        "session": str(session.session),
        "n_units_modelled": int(len(refined_ids)),
        "glm_median_r2_dev_cv": float(glm.table["r2_dev_cv"].median()),
        "glm_n_significant": int(glm.table["significant"].sum()),
        "glm_sta_median_corr": (float(agreement["kernel_sta_corr"].median())
                                if len(agreement) else np.nan),
        "rrr_best_rank": int(rrr.loc[rrr["r2_cv"].idxmax(), "rank"])
        if rrr["r2_cv"].notna().any() else -1,
        "rrr_best_cv_r2": float(rrr["r2_cv"].max()),
        "rrr_full_r2": float(rrr.attrs.get("r2_full", np.nan)),
        "fa_total_explained": factor.total_explained,
        "var1_r2": var["r2"],
        "var1_max_eigen_magnitude": float(eigenvalues["magnitude"].max()),
        "dfm_best_k_bic": _bic_best_k(dim_selection),
    }])
    write_table(summary, out_dir / "model_summary.csv", verbose=verbose)

    write_manifest(out_dir / "manifest.json", "model", cfg,
                   {"session": session.session})
    return ModelResult(glm=glm, rrr=rrr, factor=factor, var=var,
                       eigenvalues=eigenvalues, dim_selection=dim_selection,
                       agreement=agreement, summary=summary)


def _bic_best_k(dim_selection: pd.DataFrame) -> int:
    """BIC-optimal number of dynamic factors among *converged* fits.

    Returns ``-1`` when no fit produced a usable BIC, so the summary table has
    an explicit "not estimated" value rather than a misleading optimum.
    """
    ok = dim_selection.dropna(subset=["bic"])
    if "converged" in ok.columns:
        ok = ok[ok["converged"].astype(bool)]
    if ok.empty:
        return -1
    return int(ok.loc[ok["bic"].idxmin(), "n_factors"])


def _concatenate_conditions(part2: Part2Result):
    """Condition-averaged population series for the dynamic factor model.

    The trial-averaged trajectories are concatenated across conditions, which
    gives a single multivariate time series over all units. This keeps the
    dynamic factor model comparable to the trajectory analysis, at the cost of
    treating condition boundaries as real time (documented caveat, not a hidden
    one).
    """
    _, means = _condition_tensor(part2)
    n_cond, n_bins, n_units = means.shape
    return means.reshape(n_cond * n_bins, n_units)


def _condition_tensor(part2: Part2Result):
    from .responses import binned_response_tensor, condition_average

    edges, tensor = binned_response_tensor(
        part2.session, part2.refined_ids,
        part2.grating.loc[part2.grating["included"].values
                          & (part2.grating["abs_contrast"].values > 0), "onset"].values,
        part2.cfg.traj_window, part2.cfg.traj_bin,
    )
    labels, means, _ = condition_average(tensor, part2.conditions,
                                         part2.cfg.min_condition_trials)
    return labels, means


# ---------------------------------------------------------------------------
# Stage 3: validation
# ---------------------------------------------------------------------------
def run_validation_stage(part2: Part2Result, out_dir,
                         cfg: AnalysisConfig | None = None,
                         n_perm_null=50, verbose=True):
    """Run the validation battery for an already-computed session."""
    cfg = cfg or AnalysisConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = validate_single_session(
        session=part2.session, units_result=part2.units, grating_df=part2.grating,
        flashes=part2.flashes, grid=part2.grid, design=part2.design,
        tuning=part2.tuning, reg=part2.reg, rf_result=part2.rf, cfg=cfg,
        n_perm_null=n_perm_null, verbose=verbose,
    )
    report.write(out_dir, stem="validation_report")
    # machine-readable per-session validation summary
    summary = pd.DataFrame([{
        "session": str(part2.session.session),
        "n_checks": len(report.checks),
        "n_fail": report.n_fail,
        "n_warn": report.n_warn,
        "failed_checks": ";".join(c.name for c in report.checks
                                  if c.status == "fail"),
    }])
    summary.to_csv(out_dir / "validation_summary.csv", index=False)
    write_manifest(out_dir / "manifest.json", "validation", cfg,
                   {"session": part2.session.session})
    return report


# ---------------------------------------------------------------------------
# Stage 4: multi-session / multi-subject
# ---------------------------------------------------------------------------
#: Columns of the multi-session metrics table.
MULTISESSION_COLUMNS = [
    "session", "session_label", "subject", "date", "number",
    "n_clusters", "n_spikes", "duration_s", "mean_firing_rate",
    "n_anatomical", "n_quality", "n_functional", "responsiveness_rate",
    "mean_visual_modulation",
    "n_grating_trials", "n_conditions", "n_flashes",
    "n_sig_regression", "median_r2_cv",
    "side_decoding_acc", "side_decoding_null_mean", "side_decoding_null_p95",
    "side_decoding_n",
    "n_sig_rf", "rf_fraction", "rf_fraction_iso",
    "pca_evr_pc1", "pca_evr_pc1_2", "discriminability",
]


def session_metrics(session, cfg: AnalysisConfig | None = None,
                    verbose=False) -> dict:
    """Compute the multi-session metric row for one loaded session."""
    cfg = cfg or AnalysisConfig()
    units = identify_visual_neurons(session, cfg, verbose=False)
    refined_ids = units.refined_ids

    grating = reconstruct_grating_stimuli(session)
    flashes, grid = reconstruct_sparse_noise(session)
    design = extract_grating_representations(grating)

    row = {
        "session": str(session.session),
        "n_clusters": session.n_clusters,
        "n_spikes": int(len(session.spike_times)),
        "duration_s": float(session.duration),
        "mean_firing_rate": float(session.mean_firing_rate),
        "n_anatomical": units.summary["n_anatomical"],
        "n_quality": units.summary["n_quality"],
        "n_functional": units.summary["n_functional"],
        "responsiveness_rate": units.summary["responsiveness_rate"],
        "mean_visual_modulation": float(
            units.units.loc[units.units["refined"], "visual_modulation"].mean()
        ) if units.summary["n_functional"] else np.nan,
        "n_grating_trials": int(grating["included"].sum()),
        "n_conditions": int(grating["condition"].nunique()),
        "n_flashes": int(len(flashes)),
    }

    if len(refined_ids) < cfg.min_units:
        row.update({c: np.nan for c in MULTISESSION_COLUMNS if c not in row})
        return row

    tuning = compute_contrast_tuning(session, refined_ids, grating, cfg)
    reg = regress_responses_on_stimulus(session, refined_ids, grating, design,
                                        cfg, X=tuning.X, verbose=False)
    use_mask = grating["included"].values & (grating["abs_contrast"].values > 0)
    acc, _, n_side, y_side, X_side = decode_side(tuning.X, grating, use_mask, cfg)
    null_mean, null_p95, _ = decode_side_null(X_side, y_side, cfg)
    rf = compute_receptive_fields(session, refined_ids, flashes, grid, cfg,
                                  verbose=False)
    rf_iso = None
    if not cfg.rf_isolated_only:
        # bias-free variant: STA restricted to temporally isolated presentations
        rf_iso = compute_receptive_fields(
            session, refined_ids, flashes, grid,
            cfg.with_overrides(rf_isolated_only=True), verbose=False,
        )
    pca = population_pca(tuning.X, tuning.conds, cfg)

    row.update({
        "n_sig_regression": int(reg["significant"].sum()),
        "median_r2_cv": float(reg["r2_cv"].median()),
        "side_decoding_acc": acc,
        "side_decoding_null_mean": null_mean,
        "side_decoding_null_p95": null_p95,
        "side_decoding_n": n_side,
        "n_sig_rf": int(rf.table["significant_perm"].sum()),
        "rf_fraction": float(rf.table["significant_perm"].mean()),
        "rf_fraction_iso": (float(rf_iso.table["significant_perm"].mean())
                            if rf_iso is not None else np.nan),
        "pca_evr_pc1": float(pca.evr[0]),
        "pca_evr_pc1_2": float(pca.cum_evr[1]) if len(pca.cum_evr) > 1 else np.nan,
        "discriminability": pca.discriminability,
    })
    for c in MULTISESSION_COLUMNS:
        row.setdefault(c, np.nan)
    return row


def _metrics_worker(args):
    """ProcessPoolExecutor entry point (must be top-level to be picklable)."""
    session_path, cfg = args
    try:
        session = load_session_alf(session_path, verbose=False)
        return session_metrics(session, cfg, verbose=False)
    except Exception as exc:  # pragma: no cover - defensive
        return {"session": str(session_path), "error": f"{exc}",
                "traceback": traceback.format_exc()}


def run_multisession(data_root, out_dir, cfg: AnalysisConfig | None = None,
                     limit=None, subjects=None, jobs=1, force=False,
                     verbose=True) -> pd.DataFrame:
    """Run the pipeline over every session and aggregate the results.

    Results are cached in ``out_dir/multisession_metrics.csv`` so a batch can be
    resumed: sessions already present are skipped unless ``force=True``.
    """
    cfg = cfg or AnalysisConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "multisession_metrics.csv"

    refs = discover_sessions(data_root)
    if subjects:
        wanted = {s.lower() for s in subjects}
        refs = [r for r in refs if r.subject.lower() in wanted]
    if limit:
        refs = refs[: int(limit)]
    if verbose:
        print(f"  Discovered {len(refs)} sessions under {data_root}")

    cached = pd.DataFrame(columns=MULTISESSION_COLUMNS)
    done = set()
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        done = set(cached["session"].astype(str))

    todo = [r for r in refs if str(r.path) not in done]
    if verbose:
        print(f"  {len(refs) - len(todo)} cached, {len(todo)} to run "
              f"(jobs={jobs})")

    rows = []
    if todo:
        if jobs and jobs > 1:
            with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
                futures = {pool.submit(_metrics_worker, (str(r.path), cfg)): r
                           for r in todo}
                for i, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    rows.append(res)
                    if verbose:
                        print(f"    [{i}/{len(todo)}] {res.get('session')} "
                              f"-> {res.get('n_functional', 'error')} units")
                    # incremental checkpoint so long batches survive interruption
                    _write_metrics(out_dir, cached, rows)
        else:
            for i, r in enumerate(todo, 1):
                if verbose:
                    print(f"    [{i}/{len(todo)}] {r.name} ...")
                res = _metrics_worker((str(r.path), cfg))
                rows.append(res)
                if verbose:
                    print(f"        refined units = {res.get('n_functional')}, "
                          f"decoding = {res.get('side_decoding_acc')}")
                _write_metrics(out_dir, cached, rows)

    table = _write_metrics(out_dir, cached, rows)
    table = _decorate_multisession(table, refs)
    # re-write with the subject/date labels attached: the cache is keyed on the
    # session path only, but the persisted artifact must be self-describing
    table.to_csv(cache_path, index=False)

    if verbose:
        print(f"\n  Multi-session table: {len(table)} sessions, "
              f"{int(table['n_functional'].notna().sum())} with a visual population")
        if len(table):
            print(f"  Refined units  : median "
                  f"{table['n_functional'].median():.0f} "
                  f"(range {table['n_functional'].min():.0f}-"
                  f"{table['n_functional'].max():.0f})")
            print(f"  Side decoding  : median "
                  f"{table['side_decoding_acc'].median():.3f}")
            print(f"  RF fraction    : median {table['rf_fraction'].median():.3f}")

    # ---- figures + summary ----
    ok = table.dropna(subset=["n_functional"])
    if len(ok):
        fig.plot_multisession_overview(ok, out_dir)
        fig.plot_multisession_relationships(ok, out_dir)
        fig.plot_multisession_by_subject(ok, out_dir)
        _write_multisession_summary(ok, out_dir)

    write_manifest(out_dir / "manifest.json", "multisession", cfg,
                   {"n_sessions": int(len(table)),
                    "data_root": str(data_root),
                    "subjects": sorted(table["subject"].unique().tolist())
                    if len(table) else []})
    return table


def _write_metrics(out_dir, cached, rows):
    new = pd.DataFrame(rows)
    # exclude empty frames before concatenating: pandas deprecates (and will
    # change) the dtype behaviour of concatenating empty/all-NA entries
    frames = [f for f in (cached, new) if len(f)]
    if not frames:
        table = pd.DataFrame(columns=MULTISESSION_COLUMNS)
    else:
        table = pd.concat(frames, ignore_index=True)
        table = table.drop_duplicates(subset=["session"], keep="last")
    table.to_csv(Path(out_dir) / "multisession_metrics.csv", index=False)
    return table


def _decorate_multisession(table, refs):
    if table.empty:
        return table
    meta = pd.DataFrame([{
        "session": str(r.path), "session_label": r.label, "subject": r.subject,
        "date": r.date, "number": r.number,
    } for r in refs])
    table = table.drop(columns=[c for c in ("session_label", "subject", "date",
                                            "number") if c in table.columns],
                       errors="ignore")
    table = table.merge(meta, on="session", how="left")
    ordered = [c for c in MULTISESSION_COLUMNS if c in table.columns]
    ordered += [c for c in table.columns if c not in ordered]
    return table.sort_values(["subject", "date", "number"]).reset_index(drop=True)[
        ordered]


def _write_multisession_summary(ok, out_dir):
    numeric = [c for c in ok.columns
               if c not in ("session", "session_label", "subject", "date",
                            "number") and pd.api.types.is_numeric_dtype(ok[c])]
    desc = ok[numeric].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc.to_csv(Path(out_dir) / "multisession_summary.csv")
    by_subject = ok.groupby("subject")[numeric].mean(numeric_only=True)
    by_subject.to_csv(Path(out_dir) / "multisession_by_subject.csv")
    return desc