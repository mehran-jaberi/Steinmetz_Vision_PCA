"""Validation and refinement of the analysis pipeline.

The README's remaining review item was "validate and refine the complete
analysis pipeline". This module turns that into executable checks. Three kinds
of evidence are produced:

1. **Data-integrity checks** - alignment of spikes/clusters/channels/trials,
   monotonic timestamps, stimulus grid geometry, contrast value sets. These
   fail loudly (``fail``) because a broken assumption invalidates the numbers
   downstream.
2. **Definitional checks** - e.g. the response matrix is compared against a
   brute-force spike count, and the GLM design is compared against the movie it
   is supposed to encode. These catch off-by-one and shifting bugs.
3. **Null-calibrated checks** - the same statistics the pipeline reports
   (responsiveness rate, decoding accuracy, receptive-field fraction) are
   recomputed on surrogate data (circularly shifted spike trains, permuted
   labels) so that "significant" can be compared with what chance produces.

The output is a :class:`ValidationReport` that is written as CSV and Markdown,
which makes the checks reproducible and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import AnalysisConfig
from .encoding import decode_side_null
from .responses import bin_spike_counts, build_response_matrix
from .stimuli import grating_use_mask, isolated_flash_mask
from .units import responsiveness
from .utils import bh_fdr, count_spikes_in_windows

#: Expected set of drifting-grating contrasts in the Steinmetz task.
EXPECTED_CONTRASTS = {0.0, 0.0625, 0.125, 0.25, 0.5, 1.0}


@dataclass
class Check:
    name: str
    status: str        # "pass" | "warn" | "fail"
    detail: str = ""
    value: object = ""  # numeric or string summary

    def __post_init__(self):
        if self.status not in {"pass", "warn", "fail"}:
            raise ValueError(f"invalid status: {self.status}")


@dataclass
class ValidationReport:
    session: str
    checks: list = field(default_factory=list)

    def add(self, name, ok, detail="", value="", warn=False):
        status = "pass" if ok else ("warn" if warn else "fail")
        self.checks.append(Check(name=name, status=status, detail=detail,
                                 value=value))

    # -- reporting ---------------------------------------------------------
    @property
    def n_fail(self):
        return sum(c.status == "fail" for c in self.checks)

    @property
    def n_warn(self):
        return sum(c.status == "warn" for c in self.checks)

    def to_frame(self):
        return pd.DataFrame(
            [{"check": c.name, "status": c.status, "value": c.value,
              "detail": c.detail} for c in self.checks]
        )

    def to_markdown(self) -> str:
        lines = [f"# Pipeline validation report", "",
                 f"**Session:** `{self.session}`", "",
                 f"**Summary:** {len(self.checks)} checks, "
                 f"{self.n_fail} failed, {self.n_warn} warnings.", "",
                 "| check | status | value | detail |",
                 "| --- | --- | --- | --- |"]
        for c in self.checks:
            icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[c.status]
            lines.append(
                f"| `{c.name}` | {icon} | {c.value} | {c.detail} |"
            )
        return "\n".join(lines) + "\n"

    def write(self, out_dir, stem="validation_report"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(out_dir / f"{stem}.csv", index=False)
        (out_dir / f"{stem}.md").write_text(self.to_markdown(), encoding="utf-8")
        return out_dir / f"{stem}.csv"


# ---------------------------------------------------------------------------
# 1. Data integrity
# ---------------------------------------------------------------------------
def check_session(session, cfg: AnalysisConfig | None = None,
                  report: ValidationReport | None = None) -> ValidationReport:
    """Integrity checks on spikes, clusters, channels and trial arrays."""
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)

    st = session.spike_times
    report.add("spikes_sorted", bool(np.all(np.diff(st) >= 0)),
               "spike times must be non-decreasing (searchsorted depends on it)",
               f"n={len(st):,}")
    report.add("spikes_non_negative", bool(st.min() >= 0), "", float(st.min()))

    in_range = bool(session.cluster_annotation.size == session.n_clusters)
    report.add("cluster_metadata_length", in_range,
               "per-cluster metadata arrays must match n_clusters",
               session.n_clusters)

    cluster_max = int(session.spike_clusters.max(initial=0))
    report.add("spike_cluster_ids_valid", cluster_max < session.n_clusters,
               "every spike must map to an existing cluster",
               f"max id = {cluster_max}")

    # peakChannel is 1-based in ALF and converted to a 0-based index at load;
    # checking the converted range catches a wrong convention outright
    peak_min = int(session.cluster_peak_channel.min(initial=0))
    peak_max = int(session.cluster_peak_channel.max(initial=0))
    n_channels = len(session.channel_region)
    report.add("peak_channels_valid",
               peak_min >= 0 and peak_max < n_channels,
               "peak channel (1-based in ALF, converted at load) must index a "
               "channel region",
               f"index range {peak_min}-{peak_max} of {n_channels} channels")

    # the peak channel must also be geometrically consistent with the unit's
    # depth: a wrong 1-based/0-based convention shifts it by one channel pitch
    # (~20 um), which is how this bug was originally detected
    if session.n_clusters and len(session.channel_positions) == n_channels:
        idx = np.clip(session.cluster_peak_channel, 0, n_channels - 1)
        # the *signed* median is the sharp test: a wrong convention shifts the
        # whole distribution by ~10 um (half a channel pitch), whereas the
        # absolute median is dominated by genuine scatter between units
        bias = session.cluster_depths - session.channel_positions[idx, 1]
        med_bias = float(np.median(bias))
        report.add("peak_channel_depth_alignment", abs(med_bias) < 5.0,
                   "unit depth must match its peak channel position; an "
                   "off-by-one channel convention shifts the median by ~10 um",
                   f"median (depth - channel position) = {med_bias:.2f} um",
                   warn=True)

    # every unit used in the analysis must have at least one spike
    empty = int(np.sum(session.n_spikes_per_cluster == 0))
    report.add("units_with_spikes", empty == 0,
               "clusters with no spikes cannot be analysed",
               f"{empty} empty of {session.n_clusters}", warn=True)

    # trial arrays
    lengths = {k: len(v) for k, v in session.trial_info.items() if k != "intervals"}
    consistent = len(set(lengths.values())) == 1
    report.add("trial_arrays_consistent", consistent,
               f"trial field lengths: {sorted(set(lengths.values()))}",
               session.n_trials)
    report.add("trial_intervals_shape",
               np.asarray(session.trial_info["intervals"]).ndim == 2,
               "intervals must be an (n_trials, 2) array")

    # trial timing sanity: stimulus must precede the go cue
    onset = np.asarray(session.trial_info["visualStim_times"], float)
    gocue = np.asarray(session.trial_info["goCue_times"], float)
    bad = int(np.sum(gocue <= onset))
    report.add("stimulus_precedes_gocue", bad == 0,
               f"{bad} trials with goCue <= stimulus onset", bad, warn=True)
    report.add("onset_times_finite", bool(np.all(np.isfinite(onset))), "",
               int(np.sum(~np.isfinite(onset))))
    return report


def check_stimuli(session, grating_df, flashes, grid,
                  cfg: AnalysisConfig | None = None,
                  report: ValidationReport | None = None) -> ValidationReport:
    """Checks on the reconstructed grating and sparse-noise stimuli."""
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)

    # --- gratings ---
    contrasts = np.concatenate([
        grating_df["contrast_left"].unique(), grating_df["contrast_right"].unique(),
    ])
    unexpected = sorted(set(np.round(contrasts, 6)) - EXPECTED_CONTRASTS)
    report.add("grating_contrast_values", len(unexpected) == 0,
               "contrasts must come from the task's contrast set",
               f"unexpected: {unexpected}" if unexpected else "ok")

    choices = set(np.unique(grating_df["response_choice"]))
    report.add("choice_values", choices.issubset({-1.0, 0.0, 1.0}),
               "response_choice in {-1, 0, 1}", sorted(choices))

    n_included = int(grating_df["included"].sum())
    report.add("included_trials_present", n_included > 20,
               "too few included trials for population analyses would be fatal",
               n_included, warn=True)

    # --- sparse noise ---
    n_pos = len(session.sparseNoise["positions"])
    n_time = len(np.asarray(session.sparseNoise["times"]).ravel())
    report.add("sparse_noise_pairing", n_pos == n_time,
               "positions and times are paired by row and must have equal length",
               f"{n_pos} vs {n_time}")

    report.add("sparse_noise_sorted", bool(np.all(np.diff(flashes["time"].values) >= 0)),
               "flash times must be strictly non-decreasing after sorting", len(flashes))

    report.add("sparse_noise_grid", grid.n_x == 9 and grid.n_y == 33,
               "expected 9 x 33 grid for this stimulus",
               f"{grid.n_x} x {grid.n_y} = {grid.n_cells}")

    occ = np.bincount(flashes["cell"].values.astype(int), minlength=grid.n_cells)
    ratio = occ.max() / max(occ.min(), 1)
    report.add("sparse_noise_occupancy_balanced", ratio < 3.0,
               "occupancy must be roughly balanced for an unbiased STA",
               f"min {occ.min()}, max {occ.max()} (ratio {ratio:.2f})", warn=True)

    # Overlapping response windows only matter for *distinct* presentation
    # times: simultaneous flashes (ties in the raw timestamp array) are a single
    # presentation event, so the ISI is measured on unique timestamps. In
    # isolated-flash mode the STA only sees presentations with no neighbour
    # inside the window, so the check is applied to that effective subset.
    times = np.sort(flashes["time"].values)
    uniq, counts_t = np.unique(times, return_counts=True)
    n_ties = int(np.sum(counts_t > 1))
    mode = "all presentations used (rf_isolated_only=False)"
    if cfg.rf_isolated_only:
        keep = isolated_flash_mask(flashes, cfg.rf_window)
        uniq = np.unique(times[keep])
        mode = (f"{len(uniq)} isolated presentations used; the STA ignores "
                f"presentations < {cfg.rf_window:g} s apart")
    isi = np.diff(uniq)
    if len(isi):
        report.add("flash_isi_ge_window", bool(isi.min() >= cfg.rf_window * 0.5),
                   "distinct presentations closer than the STA window cause "
                   "overlapping responses (simultaneous flashes are allowed)",
                   f"min ISI = {isi.min():.4f} s, {n_ties} simultaneous events, "
                   + mode,
                   warn=True)

    # every flash cell must be reachable by the grid mapping (round-trip)
    roundtrip_ok = True
    if len(flashes):
        sample = flashes.sample(min(200, len(flashes)), random_state=cfg.random_seed)
        for _, r in sample.iterrows():
            xi = int(np.where(grid.xs == r["x"])[0][0])
            yi = int(np.where(grid.ys == r["y"])[0][0])
            if yi * grid.n_x + xi != int(r["cell"]):
                roundtrip_ok = False
                break
    report.add("flash_cell_mapping_roundtrip", roundtrip_ok,
               "cell index must be invertible to (x, y) for RF maps")

    return report


# ---------------------------------------------------------------------------
# 2. Definitional / alignment checks
# ---------------------------------------------------------------------------
def check_response_alignment(session, unit_ids, align_times, window,
                             n_probe=20, cfg: AnalysisConfig | None = None,
                             report: ValidationReport | None = None):
    """Compare the vectorised response matrix against brute-force counting."""
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)

    X = build_response_matrix(session, unit_ids, align_times, window)
    rng = np.random.default_rng(cfg.random_seed)
    unit_ids = np.asarray(unit_ids, dtype=int)
    n_probe = int(min(n_probe, len(align_times) * len(unit_ids)))
    rows = rng.integers(0, len(align_times), n_probe)
    cols = rng.integers(0, len(unit_ids), n_probe)

    mismatches = 0
    for r, c in zip(rows, cols):
        s = session.unit_spike_times(int(unit_ids[c]))
        t = align_times[r]
        brute = int(np.sum((s >= t + window[0]) & (s <= t + window[1])))
        if brute != int(X[r, c]):
            mismatches += 1
    report.add("response_matrix_matches_bruteforce", mismatches == 0,
               "vectorised spike counting must equal a direct boolean count",
               f"{mismatches}/{n_probe} mismatches")

    # response matrix must be non-negative integers
    report.add("response_matrix_non_negative",
               bool(np.all(X >= 0) and np.allclose(X, np.round(X))),
               "spike counts must be non-negative integers")
    return report


def check_binning(session, unit_ids, edges, cfg: AnalysisConfig | None = None,
                  report: ValidationReport | None = None):
    """Binned time series must sum to the total spike count inside ``edges``."""
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)
    edges = np.asarray(edges, float)
    worst = 0.0
    for uid in unit_ids:
        s = session.unit_spike_times(int(uid))
        inside = s[(s >= edges[0]) & (s <= edges[-1])]
        binned = bin_spike_counts(s, edges)
        diff = abs(int(binned.sum()) - len(inside))
        worst = max(worst, diff)
    report.add("binning_conserves_spikes", worst == 0,
               "sum of binned counts must equal spikes inside the bin range",
               f"max discrepancy = {int(worst)}")
    return report


def check_glm_design(movie, edges, counts, design, target, cfg=None,
                     report: ValidationReport | None = None,
                     session=None):
    """Verify the GLM design really encodes the stimulus it claims to."""
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session if session else "")
    n_cells = movie.shape[1]
    lag_bins = 1  # the GLM design uses a one-bin stimulus lag

    ok_stim = np.allclose(design[:10, :n_cells], movie[:10])
    report.add("glm_stimulus_lag_correct", ok_stim,
               "design stimulus columns must equal the movie at t - lag",
               f"lag = {lag_bins} bin")

    ok_target = np.allclose(target, counts[lag_bins:])
    report.add("glm_target_aligned", ok_target,
               "GLM target must be the spike counts of the matching bins")

    if cfg.glm_history_bins > 0 and design.shape[1] > n_cells:
        # history term h at design row i must equal counts[i + lag_bins - h]
        expected = counts[lag_bins - 1: len(target) + lag_bins - 1]
        ok_hist = np.allclose(design[:, n_cells], expected)
        report.add("glm_history_aligned", ok_hist,
                   "history term at row i must equal counts[i + lag - h]")
    return report


def check_rf_recovery(session, unit_ids, flashes, grid, cfg=None,
                      report: ValidationReport | None = None, top_n=5):
    """Spot-check that STA maps peak inside the session's stimulus region.

    A crude but useful sanity check: the summed|z| map across units should have
    its centre of mass well inside the grid rather than piled on one edge,
    which is what happens when flashes and spike times are misaligned.
    """
    from .receptive_fields import population_rf_image, compute_receptive_fields

    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)
    res = compute_receptive_fields(session, unit_ids, flashes, grid, cfg,
                                   verbose=False)
    img = population_rf_image(res, significant_only=False)
    if img is None:
        report.add("rf_population_image", False, "no RF maps computed", warn=True)
        return report
    ny, nx = grid.n_y, grid.n_x
    # map is reshaped (n_y, n_x).T -> (n_x, n_y)
    w = np.abs(img)
    total = w.sum()
    cx = float((w * np.arange(nx)[:, None]).sum() / total)
    cy = float((w * np.arange(ny)[None, :]).sum() / total)
    inside = 0.5 <= cx <= nx - 0.5 and 0.5 <= cy <= ny - 0.5
    report.add("rf_population_image_centred", inside,
               "population RF should sit inside the grid, not on its boundary",
               f"centre of mass = ({cx:.1f}, {cy:.1f}) of ({nx}, {ny})")
    return report


# ---------------------------------------------------------------------------
# 3. Null-calibrated checks
# ---------------------------------------------------------------------------
def _circular_shift(spikes, offset, t0, period):
    return (spikes - t0 + offset) % period + t0


def check_responsiveness_null(session, candidate_ids, stim_times,
                              cfg: AnalysisConfig | None = None,
                              n_perm=50, report: ValidationReport | None = None):
    """FDR-calibration of the responsiveness test under circular shifting.

    With the stimulus alignment destroyed, the fraction of units called
    responsive should be close to the FDR level (5%). A much larger value would
    mean the responsiveness criterion mostly reflects firing-rate structure
    (e.g. slow drift), not visual drive.
    """
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    if len(candidate_ids) == 0 or len(stim_times) < 5:
        report.add("responsiveness_null_calibrated", False,
                   "not enough candidates/trials to run the null", warn=True)
        return report

    t0 = float(session.spike_times.min())
    period = float(session.spike_times.max() - t0)
    rng = np.random.default_rng(cfg.random_seed)
    rates = np.empty(n_perm)
    for i in range(n_perm):
        offset = rng.uniform(0, period)
        shifted = {
            int(u): _circular_shift(session.unit_spike_times(int(u)), offset, t0, period)
            for u in candidate_ids
        }
        pvals = np.full(len(candidate_ids), np.nan)
        for k, u in enumerate(candidate_ids):
            s = np.sort(shifted[int(u)])
            r = count_spikes_in_windows(
                s, stim_times, cfg.stim_window
            ) / (cfg.stim_window[1] - cfg.stim_window[0])
            b = count_spikes_in_windows(
                s, stim_times, cfg.base_window
            ) / (cfg.base_window[1] - cfg.base_window[0])
            if len(r) > 1 and np.any(r != b):
                try:
                    _, p = stats.ttest_rel(r, b)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
            pvals[k] = p
        ok = ~np.isnan(pvals)
        if ok.any():
            rates[i] = bh_fdr(pvals[ok], alpha=cfg.fdr_alpha).mean()
        else:
            rates[i] = np.nan

    mean_rate = float(np.nanmean(rates))
    # 95% upper bound of the null rejection rate
    ub = float(np.nanpercentile(rates, 95))
    report.add("responsiveness_null_calibrated", mean_rate <= 0.15,
               "fraction of units called responsive under circular shifting "
               "(should be close to the FDR level)",
               f"null mean {mean_rate:.3f}, p95 {ub:.3f} (alpha={cfg.fdr_alpha})",
               warn=True)
    return report


def check_decoding_null(X, grating_df, cfg: AnalysisConfig | None = None,
                        report: ValidationReport | None = None, session=None):
    """Compare side-decoding accuracy with a label-permutation null."""
    from .encoding import decode_side

    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session if session else "")
    use = grating_use_mask(grating_df)
    acc, acc_std, n, y, Xd = decode_side(
        X, grating_df, use, cfg
    )
    if not np.isfinite(acc):
        report.add("decoding_above_chance", False,
                   "decoding could not be evaluated", warn=True)
        return report

    null_mean, null_p95, null_accs = decode_side_null(Xd, y, cfg)
    p = float((1 + np.sum(null_accs >= acc)) / (len(null_accs) + 1))
    # Reported as a warn, not a failure: whether decoding beats chance is a
    # scientific result about this session, not a pipeline defect. The numbers
    # are always recorded so the result cannot be quoted without its null.
    report.add("decoding_above_chance", acc > null_p95,
               "observed accuracy vs the label-permutation null (a scientific "
               "result, reported for every session)",
               f"acc {acc:.3f}, null mean {null_mean:.3f}, p95 {null_p95:.3f}, "
               f"p = {p:.4f} (n={n})", warn=True)
    return report


def check_rf_null_calibration(rf_table, cfg: AnalysisConfig | None = None,
                              report: ValidationReport | None = None,
                              session=None):
    """The Gaussian ANOVA-style test should flag ~5% of units by construction."""
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session if session else "")
    if rf_table is None or rf_table.empty:
        report.add("rf_null_calibration", False, "no RF table", warn=True)
        return report

    frac_gauss = float(rf_table["significant"].mean())
    frac_perm = float(rf_table["significant_perm"].mean())
    report.add("rf_significance_consistent", frac_perm >= frac_gauss * 0.5,
               "permutation test is stricter; a large gap means the Gaussian "
               "threshold over-calls significance",
               f"Gaussian {frac_gauss:.2f} vs permutation {frac_perm:.2f}",
               warn=True)
    return report


def check_response_reliability(session, unit_ids, grating_df,
                               cfg: AnalysisConfig | None = None,
                               report: ValidationReport | None = None):
    """Odd/even split reliability of the stimulus-locked response per unit.

    For each unit the condition-mean response is computed on the odd trials and
    on the even trials, and the two profiles are correlated across conditions.
    This is the relevant reliability question for the downstream analyses: a
    unit whose stimulus responses do not replicate across independent trials
    cannot support tuning/decoding/RF claims. It is also a much better check
    than raw firing-rate stability, which is dominated by slow state changes.
    """
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)

    use = grating_use_mask(grating_df)
    align = grating_df.loc[use, "onset"].values
    conds = grating_df.loc[use, "condition"].values
    X = build_response_matrix(session, unit_ids, align, cfg.stim_window)

    odd = np.arange(len(align)) % 2 == 1
    corrs = []
    for j in range(X.shape[1]):
        a, b, n_cond = [], [], 0
        for c in dict.fromkeys(conds.tolist()):
            m = conds == c
            m_odd, m_even = m & odd, m & ~odd
            if m_odd.sum() < 2 or m_even.sum() < 2:
                continue
            a.append(X[m_odd, j].mean())
            b.append(X[m_even, j].mean())
            n_cond += 1
        if n_cond >= 5 and np.std(a) > 0 and np.std(b) > 0:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    if not corrs:
        report.add("response_split_half_reliability", False,
                   "could not be estimated (too few repeated conditions)",
                   warn=True)
        return report

    med = float(np.median(corrs))
    report.add("response_split_half_reliability", med > 0.2,
               "median across units of the odd/even correlation of the "
               "condition-mean response (reliability of stimulus responses)",
               f"median r = {med:.3f} (n={len(corrs)} units)", warn=False)
    return report


def check_state_coupling(session, unit_ids, cfg: AnalysisConfig | None = None,
                         report: ValidationReport | None = None,
                         n_bins=20):
    """How much of each unit's rate is explained by the global state.

    Correlation between a unit's binned firing rate and the population-mean
    rate profile. Very high values mean the unit tracks global state changes,
    which is worth knowing before interpreting responsiveness as visual drive.
    """
    cfg = cfg or AnalysisConfig()
    report = report or ValidationReport(session=session.session)
    edges = np.linspace(float(session.spike_times.min()),
                        float(session.spike_times.max()), n_bins + 1)
    profiles = np.array([bin_spike_counts(session.unit_spike_times(int(u)), edges)
                         for u in unit_ids], dtype=float)
    if profiles.size == 0 or profiles.shape[0] < 2:
        report.add("unit_state_coupling", False, "not enough units", warn=True)
        return report
    global_profile = profiles.mean(axis=0)
    if np.std(global_profile) == 0:
        report.add("unit_state_coupling", False, "constant global rate",
                   warn=True)
        return report
    corrs = []
    for p in profiles:
        if np.std(p) == 0:
            continue
        corrs.append(float(np.corrcoef(p, global_profile)[0, 1]))
    med = float(np.median(corrs)) if corrs else np.nan
    report.add("unit_state_coupling", not (np.isfinite(med) and med > 0.9),
               "median correlation between unit rate and global rate profile "
               "across the session (high values indicate state-driven units)",
               f"median r = {med:.3f} (n={len(corrs)})", warn=True)
    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def validate_single_session(session, units_result, grating_df, flashes, grid,
                            design, tuning, reg, rf_result,
                            cfg: AnalysisConfig | None = None,
                            n_perm_null=50, verbose=True) -> ValidationReport:
    """Run the full validation battery for one session."""
    cfg = cfg or AnalysisConfig()
    report = ValidationReport(session=session.session)

    check_session(session, cfg, report)
    check_stimuli(session, grating_df, flashes, grid, cfg, report)

    refined_ids = units_result.refined_ids
    align = grating_df.loc[grating_use_mask(grating_df), "onset"].values
    check_response_alignment(session, refined_ids, align, cfg.stim_window,
                             cfg=cfg, report=report)

    # GLM design alignment on one representative unit; the movie defines the
    # shared bin edges used by both the spike counts and the design matrix
    from .glm import build_stimulus_design
    from .stimuli import build_stimulus_movie

    movie_edges, movie = build_stimulus_movie(flashes, grid, bin_size=cfg.glm_bin)
    check_binning(session, refined_ids, movie_edges, cfg, report)

    uid = int(refined_ids[0]) if len(refined_ids) else 0
    counts = bin_spike_counts(session.unit_spike_times(uid), movie_edges)[
        : movie.shape[0]
    ]
    d, target, _ = build_stimulus_design(movie, movie_edges, counts, cfg)
    check_glm_design(movie, movie_edges, counts, d, target, cfg, report, session)

    # null calibration
    check_responsiveness_null(session, units_result.units.index[
        units_result.units["passes_basic"]].tolist(),
        units_result.stim_times, cfg, n_perm=n_perm_null, report=report)
    check_decoding_null(tuning.X, grating_df, cfg, report, session)
    check_rf_null_calibration(rf_result.table, cfg, report, session)
    check_response_reliability(session, refined_ids, grating_df, cfg, report)
    check_state_coupling(session, refined_ids, cfg, report)

    if verbose:
        print(f"  Validation: {len(report.checks)} checks, "
              f"{report.n_fail} failed, {report.n_warn} warnings")
        for c in report.checks:
            if c.status != "pass":
                print(f"    [{c.status.upper()}] {c.name}: {c.detail} "
                      f"({c.value})")
    return report