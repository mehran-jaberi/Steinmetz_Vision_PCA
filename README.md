# Neural Population Analysis of Visual and Behavioral Responses

> **Work in progress — preliminary analysis**

This repository contains an ongoing analysis pipeline for investigating neural population activity in electrophysiological recordings, with an initial focus on spike-train dynamics, behavioral variables, and visual stimulus responses.

The current implementation is being developed around the **Steinmetz et al. (2019)** Neuropixels dataset and uses data stored in **ALF format**. The long-term goal is to move beyond basic single-neuron analyses and examine how population-level neural activity represents sensory stimuli and behavioral variables.

## Current Status

This project is **actively under development** and now consists of two components:

1. **Part 1 — exploratory notebook** (`Main_Steinmetz.ipynb`): general session exploration covering data loading, firing-rate statistics, PSTHs, choice-related activity, population decoding, probe-anatomy visualizations, behavioral visualizations, and an initial (preliminary) identification of visual-area neurons.

2. **Part 2 — scripted vision pipeline** (`PART2.py`): a quantitative, population-level analysis of the visual component, implementing refined visual-neuron identification, visual-stimulus reconstruction, quantitative stimulus representations, stimulus–activity relationships (tuning, regression, decoding, receptive fields), and PCA / dimensionality reduction.

The analysis is intended as an evolving **prototype** rather than a finalized pipeline; parameters, preprocessing choices, statistical procedures, and visualizations are expected to change as the project develops and preliminary results inform the next stages of analysis.

## Analysis Workflow

The vision-specific workflow outlined below is now **implemented in `PART2.py`** and mirrors the long-term goal of the project: to examine whether population activity captures meaningful structure in visual stimuli and how that structure evolves over time.

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
```

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
* **Receptive fields** via spike-triggered averages on the sparse-noise grid, with Monte-Carlo significance testing.

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
| `visual_units.csv` | Per-unit identification results |
| `grating_stimuli.csv`, `stimulus_design.csv` | Reconstructed grating stimuli |
| `flashes.csv` | Sparse-noise flash sequence |
| `tuning.csv`, `regression.csv`, `receptive_fields.csv` | Stimulus–activity relationships |
| `part2_summary.csv` | Summary statistics |

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
├── PART2.py               # Part 2: vision-specific population pipeline
├── functions.py           # shared helpers (used by the notebook)
├── pyproject.toml         # uv-managed dependencies
├── config/
│   └── category_patterns.json
├── part2_outputs/         # figures + CSV tables produced by PART2.py
├── Steinmetz_et_al_2019_9974357/   # ALF-format dataset (obtained separately)
├── steinmetz-et-al-2019-master/    # original MATLAB analysis code
└── README.md
```

## Running the Analysis

```bash
# Part 1 (exploratory notebook)
uv run python -m jupyter notebook Main_Steinmetz.ipynb

# Part 2 (vision-specific pipeline; writes to part2_outputs/)
uv run python PART2.py
```

The session path is configured at the top of `PART2.py` (`SESSION`); change it to analyse a different recording. Each stage of `PART2.py` prints progress and summary statistics as it runs.

## Important Note About the Code

A substantial portion of the current implementation was developed with the assistance of **generative AI**.

At this stage, the code should therefore be regarded as a **prototype for developing and testing the analysis workflow**, rather than polished research software. The implementation will be progressively reviewed, validated, simplified, and rewritten where necessary once preliminary results are available.

In particular, future revisions will focus on:

* Verifying analysis assumptions
* Checking preprocessing decisions
* Validating statistical procedures
* Removing unnecessary or redundant code
* Improving computational efficiency
* Making analysis steps reproducible
* Separating exploratory code from finalized analysis
* Documenting methodological decisions based on the resulting data

The scientific interpretation of the results will not be based solely on the output of the current implementation.

## Project Goals

The broader goal of the project is to move from **single-neuron response analysis toward population-level representations of sensory information**.

In particular, the project will explore questions such as:

* How are visual stimuli represented across populations of neurons?
* How does stimulus-related activity evolve over time?
* Which aspects of the stimulus are reflected in neural population activity?
* Can low-dimensional representations capture meaningful structure in the recorded population?
* How do sensory representations relate to behavioral variables and decisions?

The PCA-based analysis implemented in `PART2.py` is the first step toward investigating these population-level representations; follow-up work will extend it to richer models of population dynamics.

## Status

**Current stage:** Exploratory / preliminary analysis

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
* [ ] Validate and refine the complete analysis pipeline
* [ ] Multi-session / multi-subject analysis
* [ ] Model-based population analysis (e.g. GLM, latent dynamics)

---

### References

Steinmetz, N. A., et al. (2019). *Distributed coding of choice, action and engagement across the mouse brain*. **Nature**, 576, 266–273.

The dataset and associated resources should be obtained from the original data repository rather than included directly in this repository.
