# Neural Population Analysis of Visual and Behavioral Responses

> **Work in progress — preliminary analysis**

This repository contains an ongoing analysis pipeline for investigating neural population activity in electrophysiological recordings, with an initial focus on spike-train dynamics, behavioral variables, and visual stimulus responses.

The current implementation is being developed around the **Steinmetz et al. (2019)** Neuropixels dataset and uses data stored in **ALF format**. The long-term goal is to move beyond basic single-neuron analyses and examine how population-level neural activity represents sensory stimuli and behavioral variables.

## Current Status

This project is **actively under development** and now consists of three components:

1. **Part 1 — exploratory notebook** (`Main_Steinmetz.ipynb`): general session exploration covering data loading, firing-rate statistics, PSTHs, choice-related activity, population decoding, probe-anatomy visualizations, behavioral visualizations, and an initial (preliminary) identification of visual-area neurons.

2. **Part 2 — single-session vision pipeline** (`vision_pipeline/`, run via `PART2.py`): a quantitative, population-level analysis of the visual component, implementing refined visual-neuron identification, visual-stimulus reconstruction, quantitative stimulus representations, stimulus–activity relationships (tuning, regression, decoding, receptive fields), and PCA / dimensionality reduction. `PART2.py` is now a thin backwards-compatible entry point; the analysis itself lives in the package.

3. **Combined pipeline** (`combined_pipeline.py`): a staged CLI around the same package that adds the three stages the README previously listed as future work:
   * **validation** — executable integrity / alignment / null-calibration checks on a session (`validation_outputs/`);
   * **model** — Poisson GLM encoding models, reduced-rank regression and latent-dynamics models (`model_outputs/`);
   * **multisession** — the same metric pipeline over every available session, cached and parallelisable (`multisession_outputs/`).

The analysis is intended as an evolving **prototype** rather than a finalized pipeline; parameters, preprocessing choices, statistical procedures, and visualizations are expected to change as the project develops and preliminary results inform the next stages of analysis. Methodological caveats that the validation stage surfaced are documented explicitly under [Methodological findings and limitations](#methodological-findings-and-limitations).

## Analysis Workflow

The vision-specific workflow outlined below is now **implemented in the `vision_pipeline` package** (run via `PART2.py`, or `combined_pipeline.py --stages single`) and mirrors the long-term goal of the project: to examine whether population activity captures meaningful structure in visual stimuli and how that structure evolves over time.

```text
Visual stimulus information
          │
          ▼
Reconstruct stimulus presentation
          │
          ▼
Extract visual features / representations
          │
          ▼
Relate stimulus features to neural activity
          │
          ▼
Population-level representation
          │
          ▼
PCA / dimensionality reduction
          │
          ▼
Investigate neural population dynamics
          │
          ├──────────────► Validate the pipeline (integrity · alignment · nulls)
          │
          ├──────────────► Model-based encoding (Poisson GLM · reduced-rank regression)
          │
          └──────────────► Multi-session / multi-subject aggregation
```

All three follow-up branches are implemented in `combined_pipeline.py`; see
[Part 3](#part-3--validation-model-based-and-multi-session-stages).

## Part 1 — Exploratory Analysis (`Main_Steinmetz.ipynb`)

The exploratory notebook currently implements:

### 1. Data loading

The notebook includes utilities for loading electrophysiological recordings and associated trial information, including:

* Spike times
* Spike cluster assignments
* Cluster metadata
* Probe/channel positions
* Neuron depths
* Waveform information
* Trial events
* Visual stimulus parameters
* Behavioral variables

The current implementation primarily works with **ALF-formatted data**.

### 2. Basic neural statistics

Initial exploratory analyses include:

* Recording duration
* Number of recorded units
* Total spike count
* Mean and median firing rates
* Firing-rate distributions
* Firing rate as a function of recording depth

### 3. PSTH analysis

Peri-stimulus time histograms are used to examine neural responses relative to behavioral and sensory events.

Current analyses include alignment to:

* Go cues
* Visual stimulus onset

PSTHs can also be calculated for subsets of trials, such as trials associated with different behavioral choices.

### 4. Choice-related activity

The current pipeline examines differences in neural firing associated with left versus right choices.

This includes:

* Choice-conditioned firing rates
* Choice modulation indices
* Spike rasters
* Population PSTHs
* Choice selectivity as a function of probe depth

### 5. Population decoding

A preliminary logistic-regression decoder is used to test whether population spike counts contain information about behavioral choice.

Cross-validation is used to estimate decoding accuracy, with comparison against a pre-event baseline window.

This is intended as an initial test of whether choice-related information can be detected at the population level, rather than as a finalized decoding analysis.

### 6. Population activity visualization

Several population-level visualizations are currently implemented, including:

* Population PSTH heatmaps
* Neurons sorted by response latency
* Probe anatomy maps
* Neural selectivity across recording depth
* Firing rate versus depth

### 7. Behavioral analysis

The notebook currently includes exploratory visualizations of:

* Reaction times
* Reaction time across the recording session
* Reaction time versus stimulus contrast
* Wheel movement
* Lick timing
* Trial outcomes

### 8. Preliminary visual-neuron analysis

The latest part of the notebook begins to identify neurons recorded from visual regions based on probe/region information.

The current implementation explores:

* Identification of visual probes
* Assignment of neurons to recording regions
* Visual stimulus-aligned PSTHs
* Baseline versus stimulus-period firing
* Visual modulation
* Basic visual response metrics

This section is still particularly preliminary and is expected to change substantially as the vision-specific analysis is developed.

## Part 2 — Vision-Specific Population Analysis (`PART2.py`)

`PART2.py` implements the quantitative, population-level treatment of the visual component of the data. It is a standalone, self-contained script (it loads the session itself and does not depend on notebook state) that processes the example session `Cori / 2016-12-14 / 001`.

### 1. Refined visual-neuron identification

A three-stage cascade is used to isolate visual neurons:

* **Anatomical** — every unit is mapped to a brain region through its peak recording channel using `channels.brainLocation.tsv` (Allen ontology) and restricted to visual areas (`VIS*`).
* **Quality** — only well-isolated units are kept (`clusters._phy_annotation == 1`, i.e. good units).
* **Functional** — units must be significantly driven by the task-grating stimulus onset (paired t-test across stimulus trials, Benjamini–Hochberg FDR correction).

For the example session this yields 220 visual-area units → 42 good-quality → 34 visually responsive units.

### 2. Visual stimulus reconstruction

* **Task gratings** — per-trial stimulus presentations are reconstructed (onset/offset times, left/right contrast, signed contrast, condition labels). Note that bilateral and blank trials are present in this dataset.
* **Sparse noise** — the 9 × 33 position grid (297 cells) and flash onset times are reconstructed. The raw timestamp array is unsorted, so flashes are sorted jointly with their positions before analysis.

### 3. Quantitative stimulus representations

* Per-trial grating **design matrix** (left contrast, right contrast, interaction, signed contrast, absolute contrast).
* Sparse-noise **flash sequence**, a time-binned **stimulus movie**, and per-cell **occupancy** statistics.

### 4. Relating stimulus representations to neural activity

* **Trial × unit response matrices** (spike counts in a 0.05–0.35 s window after stimulus onset).
* **Contrast tuning curves** and a population **tuning surface** over (left, right) contrast.
* **Per-neuron linear regression** of responses on stimulus features (R², F-test p-values, side preference).
* **Population decoding** of stimulus side (cross-validated logistic regression on unilateral trials).
* **Receptive fields** via spike-triggered averages on the sparse-noise grid, with Monte-Carlo significance testing. The sparse-noise sequence runs at a minimum inter-flash interval (~10 ms) that is far shorter than the 100 ms STA window, so the full-sequence average mixes responses to neighbouring flashes. The pipeline therefore also computes an **isolated-flash** variant (only presentations with no other presentation inside the window, `rf_isolated_only`) and stores it in the same table under `*_iso` columns — see [Methodological findings and limitations](#methodological-findings-and-limitations).

For the example session: 8/34 units significantly encode contrast, side decoding reaches ~59% (n = 87 unilateral trials), and 31/34 refined units have significant receptive fields — a strong functional confirmation of the visual identification.

### 5. PCA / dimensionality reduction

* PCA of the trial × unit response matrix (scree plot, explained variance).
* Condition structure in PC space (PC scatter, within-vs-across condition distance ratio / discriminability).
* **Time-resolved population trajectories** — condition-averaged responses binned over time, projected into PC space.

For the example session: PC1 explains ~28% of the variance, PC1–2 ~35%, and condition discriminability is low, consistent with a brief response window and largely gain-like contrast encoding in V1.

### Outputs

All figures (PNG) and result tables (CSV) are written to `part2_outputs/`:

| Output | Contents |
| --- | --- |
| `fig01_visual_neuron_refinement.png` | Refinement cascade and region composition |
| `fig01b_responsiveness_vs_depth.png` | Responsiveness vs probe depth |
| `fig03_stimulus_representations.png` | Design matrix, occupancy, flash sequence |
| `fig04_contrast_tuning.png` | Population tuning + tuning surface |
| `fig04b_stimulus_regression.png` | Encoding strength, side preference vs depth |
| `fig04c_receptive_fields.png`, `fig04d_rf_fraction.png` | Example RFs and significance |
| `fig05_pca.png`, `fig05b_time_resolved_pca.png` | PCA structure and trajectories |
| `visual_units.csv` | Per-unit identification results (region, depth, peak channel, refinement flags) |
| `grating_stimuli.csv`, `stimulus_design.csv` | Reconstructed grating stimuli |
| `flashes.csv` | Sparse-noise flash sequence |
| `tuning.csv`, `regression.csv`, `receptive_fields.csv` | Stimulus–activity relationships (`receptive_fields.csv` carries the `*_iso` isolated-flash columns) |
| `part2_summary.csv` | Summary statistics |
| `manifest.json` | Config, library versions and git revision for the run |

## Part 3 — Validation, Model-Based and Multi-Session Stages (`combined_pipeline.py`)

The three items the README previously listed as future work are implemented as independent stages of one CLI. Each stage writes its own `manifest.json` (configuration, library versions, git revision) so a result can be traced back to the exact code and parameters that produced it.

### Stage 7 — validation (`validation_outputs/`)

`vision_pipeline/validation.py` turns "validate the pipeline" into 31 executable checks in three families:

* **data integrity** — spikes sorted / non-negative, cluster metadata lengths, spike→cluster ids, peak-channel range and *geometric* consistency, trial field lengths, stimulus grid geometry, contrast value set, occupancy balance;
* **definitional / alignment** — the vectorised response matrix is compared against a brute-force spike count, binned counts against the spike total, and the GLM design against the stimulus movie it is meant to encode (this catches off-by-one and shifting bugs);
* **null-calibrated** — responsiveness rate, decoding accuracy and receptive-field fractions are recomputed on surrogate data (circularly shifted spike trains, permuted labels) so "significant" can be compared with what chance produces.

For `Cori / 2016-12-14 / 001`: **31 checks, 0 failed, 2 warnings**. Both warnings are reported for every session rather than treated as errors, because they describe the data, not the code: the sparse-noise inter-flash interval (~10 ms, 749 simultaneous presentations) and the fact that side decoding is only marginally above its permutation null (58.9% vs 50.3%, p = 0.09, n = 87).

Outputs: `validation_report.md` / `.csv` (per-check table), `validation_summary.csv` (per-session summary), `manifest.json`.

### Stage 6 — model-based population analysis (`model_outputs/`)

* **Poisson GLM encoding models** of each refined unit on the sparse-noise movie (stimulus + spike-history terms, L2 penalty), fitted per unit with held-out deviance explained and an FDR-corrected stimulus-drive test. For the example session: 34/34 units fitted, **13/34 significantly stimulus-driven**, median held-out deviance explained 0.013.
* **GLM vs STA agreement** — correlation between the GLM's stimulus kernel and the STA map: a model-based check of the receptive-field analysis.
* **Reduced-rank regression** of the population time series on the stimulus movie (the same model family as the MATLAB `reducedRankRegression` code shipped with the dataset): best held-out rank 2 (CV R² = 0.073; full-rank in-sample R² = 0.157).
* **Latent dynamics** — factor analysis of the trial × unit response matrix (3 factors, 34.9% of total response variance), a VAR(1) on the condition-averaged trajectories (one-step R² = 0.43, max |eigenvalue| = 0.77), and a dynamic factor model with BIC selection over the number of factors.

The dynamic factor model is the least stable component: the ML fit converges for only 2 of the 5 candidate dimensions on the example session, so BIC is taken among converged fits only and the convergence count is reported (and stored per dimension). This is a documented limitation, not a silently ignored one.

Outputs: `glm_encoding.csv`, `glm_vs_sta.csv`, `rrr_rank_curve.csv`, `latent_factor_loadings.csv`, `latent_var1_eigenvalues.csv`, `latent_dimension_selection.csv`, `model_summary.csv`, figures `fig06*` / `fig07*`, `manifest.json`.

### Stage 8 — multi-session / multi-subject analysis (`multisession_outputs/`)

Runs the whole metric pipeline (identification → stimulus reconstruction → tuning → regression → decoding → receptive fields → PCA) over every session found under the data root, in parallel worker processes, with an incremental cache (`multisession_metrics.csv`) so a long batch can be resumed; `--force` recomputes everything.

All **39 sessions of the local dataset** (10 subjects) run without errors, and **24 sessions have at least one refined visual unit** (median 8, max 76; the remainder have no probe in visual cortex or too few well-isolated units). Median across sessions: side-decoding accuracy 0.68, significant-RF fraction 0.19, PC1 variance 0.16.

Outputs: `multisession_metrics.csv` (one row per session, with subject/date labels), `multisession_summary.csv`, `multisession_by_subject.csv`, figures `fig10`–`fig12`, `manifest.json`.

### Methodological findings and limitations

* **The peak-channel convention had an off-by-one bug, now fixed.** `clusters.peakChannel` is 1-based in ALF while the `channels.*` tables are 0-based. Indexing them directly (the original behaviour) shifted every unit's channel — and therefore its brain-region label — by one channel (~10 µm), and crashed outright on the 8 sessions where the maximum channel id equalled the channel count. The loader now converts once; the fix was verified against the probe geometry (median unit depth − channel position: 0.3 µm after conversion vs −10.5 µm before) and is guarded by a new validation check that fails when the convention is wrong.
* **Receptive-field significance depends on stimulus overlap.** The sparse-noise sequence has a minimum inter-flash interval of ~10 ms against a 100 ms STA window, so only ~10–25% of presentations are temporally isolated. Restricting the STA to those leaves 764/7962 presentations (example session) or 346/12705 (largest session), and **no unit survives FDR correction in any of the 39 sessions** under that test, whereas the full-sequence STA calls 28/34 (example session) or 22/67 (largest session) significant. The full-sequence result is kept as the primary analysis for comparability, but this means the STA-based receptive-field evidence cannot be separated from overlap contamination at this stimulus timing — the GLM stage, which models the full stimulus history, is the appropriate tool for RF estimation here.
* **Side decoding is marginal.** The median across sessions is 0.68, but for the example session 58.9% is only at the edge of the permutation null (p ≈ 0.09, n = 87 unilateral trials); the validation stage reports this for every session instead of asserting significance.
* **Dynamic-factor fits frequently fail to converge** (see Stage 6); BIC selection is therefore restricted to converged fits and reported with the convergence count.
* **The dynamic factor model treats condition boundaries as real time** — condition-averaged trajectories are concatenated across conditions, which is a deliberate, documented simplification.

## Dataset

The analysis is currently being developed using the **Steinmetz et al. (2019) Neuropixels dataset**, which contains large-scale electrophysiological recordings together with behavioral and stimulus information.

The repository does **not** contain the raw dataset. Users should obtain the dataset separately and configure the appropriate local data path before running the analysis (see `SESSION` in `PART2.py` and the data paths in `Main_Steinmetz.ipynb`).

## Requirements

The analysis relies primarily on Python scientific-computing and machine-learning libraries, including:

```text
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
statsmodels
```

Dependencies are managed with **uv** via `pyproject.toml`. Note that the base Anaconda environment on this machine has a broken NumPy/pandas stack, so run everything through the project environment, e.g. `uv run python PART2.py` or `uv run python -m jupyter notebook`.

## Repository Structure

```text
.
├── Main_Steinmetz.ipynb   # Part 1: exploratory notebook
├── vision_pipeline/       # Part 2/3: the analysis package (15 modules)
│   ├── config.py          # AnalysisConfig — all tunable parameters
│   ├── io.py              # ALF loading, session discovery
│   ├── stimuli.py         # grating / sparse-noise reconstruction
│   ├── units.py           # refined visual-neuron identification
│   ├── responses.py       # response matrices, PSTHs, binned tensors
│   ├── encoding.py        # contrast tuning, regression, decoding
│   ├── receptive_fields.py # spike-triggered-average receptive fields
│   ├── population.py      # PCA and condition structure
│   ├── glm.py             # Poisson stimulus encoding models
│   ├── latent.py          # reduced-rank regression, latent dynamics
│   ├── validation.py      # integrity, alignment and null-calibration checks
│   ├── figures.py         # all plots
│   ├── pipeline.py        # stage orchestration
│   └── reporting.py       # run manifests (config, versions, git revision)
├── PART2.py               # thin entry point for the single-session stage
├── combined_pipeline.py   # staged CLI: single · validate · model · multisession
├── functions.py           # shared helpers (used by the notebook)
├── pyproject.toml         # uv-managed dependencies
├── config/
│   └── category_patterns.json
├── part2_outputs/         # single-session figures + tables
├── model_outputs/         # GLM / reduced-rank / latent-dynamics results
├── validation_outputs/    # per-session validation report
├── multisession_outputs/  # cross-session tables + figures
├── Steinmetz_et_al_2019_9974357/   # ALF-format dataset (obtained separately)
├── steinmetz-et-al-2019-master/    # original MATLAB analysis code
└── README.md
```

## Running the Analysis

```bash
# Part 1 (exploratory notebook)
uv run python -m jupyter notebook Main_Steinmetz.ipynb

# Part 2 (single-session vision pipeline; writes to part2_outputs/)
uv run python PART2.py

# Full pipeline: single-session + validation + model (default stages)
uv run python combined_pipeline.py

# Chosen stages only
uv run python combined_pipeline.py --stages single,validate

# Multi-session run over the first six sessions, four processes, resumable cache
uv run python combined_pipeline.py --stages multisession --limit 6 --jobs 4

# Two subjects only, ignoring the cache
uv run python combined_pipeline.py --stages multisession --subjects Cori,Forssmann --force

# Override any AnalysisConfig field
uv run python combined_pipeline.py --stages single --set fdr_alpha=0.01 --set rf_isolated_only=true
```

The session path is configured at the top of `PART2.py` (`SESSION`) or with `--session`; the multi-session stage scans `--data-root` (default: `Steinmetz_et_al_2019_9974357/nicklab/Subjects`). Each stage prints progress and summary statistics as it runs and writes a `manifest.json` recording the configuration, library versions and git revision that produced it. The multi-session stage caches per-session metrics in `multisession_outputs/multisession_metrics.csv`, so interrupted batches resume where they stopped.

## Important Note About the Code

A substantial portion of the current implementation was developed with the assistance of **generative AI**.

The code remains a **prototype for developing and testing the analysis workflow**, but the review pass has begun: the analysis now lives in a documented package, the pipeline is exercised by 31 executable validation checks, and the decisions it depends on (peak-channel convention, response windows, isolated-flash STA, latent-model convergence) are recorded in code and summarised under [Methodological findings and limitations](#methodological-findings-and-limitations).

Status of the original review list:

* Verifying analysis assumptions — *in progress*: integrity and geometric checks in the validation stage
* Checking preprocessing decisions — *in progress*: ALF conventions audited (one indexing bug found and fixed)
* Validating statistical procedures — *in progress*: null-calibrated checks for responsiveness, decoding and receptive fields
* Removing unnecessary or redundant code — *partly done*: the ~2,000-line script is now 15 focused modules; scratch scripts (`_debug_model.py`, `_inspect_sessions.py`, `_compare_baseline.py`) still sit in the repository root
* Improving computational efficiency — *done for the batch case*: multi-session runs are cached and parallel (39 sessions in ~2 min on 4 processes)
* Making analysis steps reproducible — *done*: immutable configuration, seeded RNGs, per-stage manifests, resumable caches
* Separating exploratory code from finalized analysis — *done*: notebook (exploration) vs package + CLI (analysis)
* Documenting methodological decisions based on the resulting data — *done*: see the findings section above

The scientific interpretation of the results will not be based solely on the output of the current implementation.

## Project Goals

The broader goal of the project is to move from **single-neuron response analysis toward population-level representations of sensory information**.

In particular, the project will explore questions such as:

* How are visual stimuli represented across populations of neurons?
* How does stimulus-related activity evolve over time?
* Which aspects of the stimulus are reflected in neural population activity?
* Can low-dimensional representations capture meaningful structure in the recorded population?
* How do sensory representations relate to behavioral variables and decisions?

The PCA-based analysis (Part 2) was the first step toward these population-level representations. The pipeline now also fits richer models of the population — reduced-rank regression, factor analysis, VAR(1) and a dynamic factor model (Part 3) — so the remaining scientific work is interpretation: relating the latent structure to the stimulus and behavioural variables, and deciding which of the documented methodological corrections should become the primary analysis.

## Status

**Current stage:** All roadmap stages implemented and validated; interpretation of the population-level results in progress

**Completed or partially implemented:**

* [x] ALF data loading
* [x] Spike and cluster inspection
* [x] Basic firing-rate analysis
* [x] PSTH analysis
* [x] Choice-related analysis
* [x] Preliminary population decoding
* [x] Neural population visualizations
* [x] Behavioral visualizations
* [x] Refine visual-neuron identification
* [x] Reconstruct visual stimuli
* [x] Extract quantitative visual stimulus representations
* [x] Relate stimulus representations to neural activity
* [x] PCA / dimensionality reduction
* [x] Validate and refine the complete analysis pipeline (31 checks; two data-driven warnings documented)
* [x] Multi-session / multi-subject analysis (39 sessions, 10 subjects; cached and parallel)
* [x] Model-based population analysis (Poisson GLM, reduced-rank regression, factor analysis, VAR(1), dynamic factor model)
* [ ] Experimental-algorithm branch (`experimental_algorithm`) — intentionally untouched; to be specified separately

---

### References

Steinmetz, N. A., et al. (2019). *Distributed coding of choice, action and engagement across the mouse brain*. **Nature**, 576, 266–273.

The dataset and associated resources should be obtained from the original data repository rather than included directly in this repository.
