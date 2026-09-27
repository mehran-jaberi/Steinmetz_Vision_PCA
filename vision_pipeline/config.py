"""Configuration objects and constants for the vision population pipeline.

All tunable analysis parameters live in :class:`AnalysisConfig`. Instances are
immutable (frozen) so that a configuration object can be safely shared across
processes and recorded verbatim in run manifests, which is what makes a run
reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "Steinmetz_et_al_2019_9974357" / "nicklab" / "Subjects"
)
DEFAULT_SESSION = DEFAULT_DATA_ROOT / "Cori" / "2016-12-14" / "001"

# Legacy Part-2 output directory (kept for backwards compatibility).
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "part2_outputs"
MULTISESSION_OUTPUT_DIR = PROJECT_ROOT / "multisession_outputs"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "model_outputs"
VALIDATION_OUTPUT_DIR = PROJECT_ROOT / "validation_outputs"

# ---------------------------------------------------------------------------
# Biological constants
# ---------------------------------------------------------------------------
#: Allen ontology abbreviations treated as visual cortex.
VISUAL_AREAS = (
    "VIS", "VISa", "VISal", "VISam", "VISl", "VISli", "VISmma", "VISmmp",
    "VISp", "VISpl", "VISpm", "VISpor", "VISrl",
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable bundle of analysis parameters.

    Defaults reproduce the original ``PART2.py`` behaviour so that the
    refactored pipeline can be validated against the published outputs.
    """

    # --- response windows (seconds relative to stimulus onset) ---
    stim_window: tuple[float, float] = (0.05, 0.35)
    base_window: tuple[float, float] = (-0.35, -0.05)

    # --- unit selection ---
    good_annotation: int = 1          # phy: 1 = good, 2 = MUA, 3 = noise
    min_total_spikes: int = 50
    fdr_alpha: float = 0.05

    # --- receptive fields ---
    rf_window: float = 0.1             # spike-triggered-average integration (s)
    rf_null_samples: int = 1000        # Gaussian Monte-Carlo samples (legacy test)
    rf_permutations: int = 200         # circular-shift permutations (empirical null)

    # --- population / PCA ---
    n_pca_components: int = 10
    traj_window: tuple[float, float] = (-0.1, 0.5)
    traj_bin: float = 0.02
    movie_bin: float = 0.1
    min_condition_trials: int = 3

    # --- model-based analyses ---
    glm_bin: float = 0.1               # bin size for sparse-noise GLM
    glm_alpha: float = 1.0             # L2 penalty for the Poisson GLM
    glm_history_bins: int = 1          # number of autoregressive history bins
    rrr_ranks: tuple[int, ...] = (1, 2, 3, 5, 8, 12)
    latent_factors: tuple[int, ...] = (1, 2, 3, 4, 6)
    var_lags: int = 1

    # --- decoding / resampling ---
    cv_folds: int = 5
    n_permutations: int = 200          # label permutations for null distributions
    random_seed: int = 42

    # --- guards ---
    min_units: int = 5

    def to_dict(self) -> dict:
        """JSON/CSV-friendly representation (tuples become lists)."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, tuple):
                d[key] = list(value)
        return d

    def with_overrides(self, **kwargs) -> "AnalysisConfig":
        """Return a copy with the given fields replaced."""
        valid = {k: v for k, v in kwargs.items() if v is not None}
        for key, value in valid.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config field: {key}")
            current = getattr(self, key)
            # allow lists to be passed for tuple-typed fields
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
        return replace(self, **valid)
